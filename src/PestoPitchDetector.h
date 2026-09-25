#pragma once

#include <atomic>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <anira/anira.h>
#include <PestoModelData.h>

/**
 * PESTO pitch detector обёртка поверх ANIRA + ONNX Runtime (streaming-режим).
 *
 * Модель `models/pesto.onnx` экспортируется скриптом `ml/pesto/export_onnx.py`
 * через `load_model(streaming=True, mirror=1.0)` + обёртка `StatelessPESTO`.
 * См. ROADMAP.md → Шаг 2.5 и backlog «finetune PESTO под realtime-режим».
 *
 * Ключевые отличия от предыдущей offline-реализации:
 *
 *  1. **Streaming CQT.** Модель — это `CachedConv1d` внутри `StreamingCQT`, левый
 *     паддинг свёртки = реальные прошлые сэмплы (cache), правый край = zeros
 *     (mirror_fn=zeros, фракция mirror=0.8). Никаких reflect-pad артефактов.
 *
 *  2. **Один фрейм на вызов** (вместо 26). На каждый hop=441 сэмплов получаем
 *     ровно один F0/conf. ~25× меньше CPU.
 *
 *  3. **Cache state в C++.** ONNX Runtime stateless → выносим cache наружу как
 *     явный input/output тензор. Храним в `PestoProcessor::cache_state`,
 *     подкладываем в pre_process, читаем cache_out в post_process.
 *
 * ANIRA конфиг:
 *   inputs:   [0] audio  streamable     (1, kHopSamples)
 *             [1] cache  non-streamable (1, kCacheSize)
 *   outputs:  [0] f0_hz       non-streamable (1, 1)
 *             [1] confidence  non-streamable (1, 1)
 *             [2] volume      non-streamable (1, 1)
 *             [3] activations non-streamable (1, 1, kActivationsBins)
 *             [4] cache_out   non-streamable (1, kCacheSize)
 *
 * Бюджет задержки «струна → бас в наушниках» с Reaper buffer 256:
 *   ADC + analog               ~  2 мс
 *   DAW round-trip (in+out)    ~ 12 мс
 *   PESTO chunk accumulation   ~  5 мс (avg, max 10)
 *   PESTO mirror algorithmic    0 мс
 *   PESTO compute + worker     ~  3 мс
 *   DAC + analog               ~  3 мс
 *   ─────────────────────────────────
 *   Total                      ~ 25 мс
 *
 * API мимикрирует YinPitchDetector: `process(buf, n)` возвращает F0 (Гц) или 0.0f
 * пока валидного питча ещё нет (initial buffering или unvoiced).
 */
class PestoPitchDetector
{
public:
    // ── параметры модели (должны совпадать с pesto.onnx + pesto_onnx_meta.json) ──
    static constexpr int          kSampleRate        = 44100;
    static constexpr int          kHopSamples        = 441;    // 10 мс — chunk на вызов
    static constexpr int          kCacheSize         = 3876;   // mirror=1.0: (8192 - 441) / 2
    static constexpr int          kMirrorLagSamples  = 0;      // mirror=1.0: zero algorithmic look-back
    static constexpr int          kActivationsBins   = 384;    // output_dim Resnet1d
    static constexpr int          kFramesPerCall     = 1;      // один фрейм на inference
    static constexpr float        kVoicedThreshold   = 0.5f;   // PESTO confidence порог
    static constexpr float        kMaxInferenceMs    = 20.0f;  // SLA на inference
    static constexpr unsigned int kWarmUp            = 2;      // прогрев

    // Output tensor indices in ANIRA postprocess (соответствуют output_names ONNX-графа)
    static constexpr size_t kOutF0    = 0;
    static constexpr size_t kOutConf  = 1;
    static constexpr size_t kOutVol   = 2;
    static constexpr size_t kOutActs  = 3;
    static constexpr size_t kOutCache = 4;

    // Input tensor indices
    static constexpr size_t kInAudio = 0;
    static constexpr size_t kInCache = 1;

    PestoPitchDetector()
        : m_config(
              { anira::ModelData(
                    reinterpret_cast<void*>(const_cast<char*>(PestoModelData::pesto_onnx)),
                    static_cast<size_t>(PestoModelData::pesto_onnxSize),
                    anira::InferenceBackend::ONNX) },
              { anira::TensorShape(
                    /* input shapes  */ { { 1, kHopSamples }, { 1, kCacheSize } },
                    /* output shapes */ { { 1, kFramesPerCall },
                                          { 1, kFramesPerCall },
                                          { 1, kFramesPerCall },
                                          { 1, kFramesPerCall, kActivationsBins },
                                          { 1, kCacheSize } },
                    anira::InferenceBackend::ONNX) },
              anira::ProcessingSpec(
                  /* preprocess_input_channels   */ { 1, 1 },
                  /* postprocess_output_channels */ { 1, 1, 1, 1, 1 },
                  /* preprocess_input_size       */ { static_cast<size_t>(kHopSamples), 0 },
                  /* postprocess_output_size     */ { 0, 0, 0, 0, 0 }
              ),
              kMaxInferenceMs,
              kWarmUp),
          m_processor(m_config),
          m_handler(m_processor, m_config)
    {
        // SessionElement::m_current_backend defaults to CUSTOM — must set ONNX explicitly
        m_handler.set_inference_backend(anira::InferenceBackend::ONNX);
    }

    void prepare(double sampleRate, int blockSize)
    {
        // Модель трассирована под 44.1 кГц; на другом SR F0 будет транспонирован.
        // Перенос на другой SR = реэкспорт ONNX с новыми CQT-ядрами.
        anira::HostConfig hc { static_cast<float>(blockSize), static_cast<float>(sampleRate) };
        m_handler.prepare(hc);
    }

    void reset()
    {
        m_handler.reset();
        m_processor.reset_cache();
        m_hasValid.store(false, std::memory_order_release);
        m_lastPitch.store(0.0f, std::memory_order_relaxed);
        m_debugCount = 0;
    }

    /** Толкаем блок аудио в инференс-пайплайн и возвращаем последний валидный F0 (Гц).
     *  Возврат 0.0f означает «питч ещё не определён» — синтез должен пропустить блок.
     *
     *  push_data() пихает аудио в RingBuffer ANIRA. Как только там накопилось
     *  kHopSamples — triggers pre_process → ONNX run → post_process в worker-треде.
     *  pop_data(nullptr, 0) дёргает new_data_request() в audio-треде, который
     *  передаёт результаты в atomic storage — оттуда читаем get_output().
     */
    float process(const float* monoInput, int numSamples)
    {
        const float* const inputCh[1] = { monoInput };
        m_handler.push_data(inputCh, static_cast<size_t>(numSamples), 0);
        m_handler.pop_data(static_cast<float* const*>(nullptr), 0, 0);

        const float conf = m_processor.get_output(kOutConf, 0);
        const float f0   = m_processor.get_output(kOutF0,   0);

        // ── DEBUG: log first 500 blocks to D:\projects\BassEmulatorVST\pesto_debug.log
        ++m_debugCount;
        if (m_debugCount <= 500 && m_debugCount % 50 == 0)
        {
            if (FILE* fp = std::fopen("D:\\projects\\BassEmulatorVST\\pesto_debug.log", "a"))
            {
                std::fprintf(fp,
                    "[%d] conf=%.4f f0=%.2f | hasValid=%d\n",
                    m_debugCount, conf, f0,
                    (int)m_hasValid.load(std::memory_order_relaxed));
                std::fclose(fp);
            }
        }

        if (conf >= kVoicedThreshold && f0 > 0.0f)
        {
            m_lastPitch.store(f0, std::memory_order_relaxed);
            m_hasValid.store(true, std::memory_order_release);
        }

        return getCurrentPitch();
    }

    /** Последний валидный F0 (Гц), или 0.0f если ещё не было ни одного voiced-кадра. */
    float getCurrentPitch() const
    {
        return m_hasValid.load(std::memory_order_acquire)
             ? m_lastPitch.load(std::memory_order_relaxed)
             : 0.0f;
    }

    /** Латентность пайплайна в сэмплах (для setLatencySamples в processor).
     *  ANIRA latency = chunk accumulation + worker queue. Mirror lag добавляется
     *  отдельно: фрейм PESTO «относится» к моменту kMirrorLagSamples назад. */
    int getLatencySamples() const
    {
        return static_cast<int>(m_handler.get_latency()) + kMirrorLagSamples;
    }

private:
    // PestoProcessor хранит cache_state как member (не RT-safe atomic — обновляется
    // только из inference-треда: pre_process читает, post_process пишет).
    struct PestoProcessor : anira::PrePostProcessor
    {
        std::vector<float> cache_state;

        explicit PestoProcessor(anira::InferenceConfig& cfg)
            : anira::PrePostProcessor(cfg), cache_state(kCacheSize, 0.0f) {}

        void pre_process(std::vector<anira::RingBuffer>& input,
                         std::vector<anira::BufferF>& output,
                         anira::InferenceBackend /*backend*/) override
        {
            // input #0: audio — забираем kHopSamples из RingBuffer
            pop_samples_from_buffer(input[kInAudio], output[kInAudio],
                                    static_cast<size_t>(kHopSamples));

            // input #1: cache — копируем member-vector в ONNX-input буфер
            float* cache_dst = output[kInCache].data();
            std::copy(cache_state.begin(), cache_state.end(), cache_dst);
        }

        void post_process(std::vector<anira::BufferF>& input,
                          std::vector<anira::RingBuffer>& /*output*/,
                          anira::InferenceBackend /*backend*/) override
        {
            // outputs 0,1,2: f0/conf/vol → atomic storage (один float каждый)
            set_output(input[kOutF0].data()[0],   kOutF0,   0);
            set_output(input[kOutConf].data()[0], kOutConf, 0);
            set_output(input[kOutVol].data()[0],  kOutVol,  0);
            // output 3 (activations) пропускаем — не используется для realtime

            // output 4: cache_out → копируем в member-vector для следующего вызова
            const float* cache_src = input[kOutCache].data();
            std::copy(cache_src, cache_src + kCacheSize, cache_state.begin());
        }

        void reset_cache()
        {
            std::fill(cache_state.begin(), cache_state.end(), 0.0f);
        }
    };

    anira::InferenceConfig  m_config;
    PestoProcessor          m_processor;
    anira::InferenceHandler m_handler;

    std::atomic<float> m_lastPitch  { 0.0f };
    std::atomic<bool>  m_hasValid   { false };
    int                m_debugCount { 0 };
};
