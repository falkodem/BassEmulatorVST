#pragma once

#include <atomic>
#include <bit>
#include <cstdint>
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
 *     явный input/output тензор. Храним в `PestoProcessor::cache_state` и
 *     передаём между последовательными инференсами в worker-потоке.
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
 * process() отдаёт каждый новый кадр ровно один раз, включая unvoiced-кадры.
 * Время удержания частоты и сброс на новой атаке контролирует плагин.
 */
class PestoPitchDetector
{
public:
    struct PitchFrame
    {
        float f0 = 0.0f;
        bool updated = false;
    };

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
              kWarmUp,
              true), // cache следующего кадра зависит от предыдущего инференса
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
        m_lastReadResult = 0;
        m_samplesPushed = 0;
        m_rejectThroughSample = 0;
    }

    /** Толкаем блок аудио в инференс-пайплайн и читаем только новый результат.
     *
     *  push_data() пихает аудио в RingBuffer ANIRA. Как только там накопилось
     *  kHopSamples — pre_process в audio-треде, inference в worker-треде.
     *  pop_data(nullptr, 0) собирает готовые кадры и вызывает post_process
     *  в audio-треде.
     */
    PitchFrame process(const float* monoInput, int numSamples, bool newOnset)
    {
        const float* const inputCh[1] = { monoInput };
        m_handler.push_data(inputCh, static_cast<size_t>(numSamples), 0);
        m_samplesPushed += static_cast<std::uint64_t>(numSamples);
        if (newOnset)
            m_rejectThroughSample = m_samplesPushed;
        m_handler.pop_data(static_cast<float* const*>(nullptr), 0, 0);

        const auto result = m_processor.latestResult.load(std::memory_order_acquire);
        if (result == m_lastReadResult)
            return {};

        m_lastReadResult = result;
        // На атаке отбрасываем все кадры, завершившиеся до конца её блока.
        const auto frameEndSample = (result >> 32) * kHopSamples;
        if (frameEndSample <= m_rejectThroughSample)
            return {};
        return { std::bit_cast<float>(static_cast<std::uint32_t>(result)), true };
    }

    /** Латентность пайплайна в сэмплах (для setLatencySamples в processor).
     *  ANIRA latency = chunk accumulation + worker queue. Mirror lag добавляется
     *  отдельно: фрейм PESTO «относится» к моменту kMirrorLagSamples назад. */
    int getLatencySamples() const
    {
        return static_cast<int>(m_handler.get_latency()) + kMirrorLagSamples;
    }

private:
    // cache_state используется только worker-потоком между вызовами модели.
    struct PestoProcessor : anira::PrePostProcessor
    {
        std::vector<float> cache_state;
        std::atomic<std::uint64_t> latestResult { 0 };
        std::uint32_t frameCounter = 0;

        explicit PestoProcessor(anira::InferenceConfig& cfg)
            : anira::PrePostProcessor(cfg), cache_state(kCacheSize, 0.0f) {}

        void pre_process(std::vector<anira::RingBuffer>& input,
                         std::vector<anira::BufferF>& output,
                         anira::InferenceBackend /*backend*/) override
        {
            // input #0: audio — забираем kHopSamples из RingBuffer
            pop_samples_from_buffer(input[kInAudio], output[kInAudio],
                                    static_cast<size_t>(kHopSamples));
        }

        void before_inference(std::vector<anira::BufferF>& input,
                              anira::InferenceBackend /*backend*/) override
        {
            std::copy(cache_state.begin(), cache_state.end(), input[kInCache].data());
        }

        void after_inference(std::vector<anira::BufferF>& output,
                             anira::InferenceBackend /*backend*/) override
        {
            const float* cache_src = output[kOutCache].data();
            std::copy(cache_src, cache_src + kCacheSize, cache_state.begin());
        }

        void post_process(std::vector<anira::BufferF>& input,
                          std::vector<anira::RingBuffer>& /*output*/,
                          anira::InferenceBackend /*backend*/) override
        {
            // Один атомарный снимок: номер кадра + F0 после voiced-гейта.
            const float confidence = input[kOutConf].data()[0];
            const float f0 = input[kOutF0].data()[0];
            const float voicedF0 = confidence >= kVoicedThreshold && f0 > 0.0f
                                 ? f0 : 0.0f;
            const auto result = (static_cast<std::uint64_t>(++frameCounter) << 32)
                              | std::bit_cast<std::uint32_t>(voicedF0);
            latestResult.store(result, std::memory_order_release);
        }

        void reset_cache()
        {
            std::fill(cache_state.begin(), cache_state.end(), 0.0f);
            frameCounter = 0;
            latestResult.store(0, std::memory_order_relaxed);
        }
    };

    anira::InferenceConfig  m_config;
    PestoProcessor          m_processor;
    anira::InferenceHandler m_handler;

    std::uint64_t m_lastReadResult = 0;
    std::uint64_t m_samplesPushed = 0;
    std::uint64_t m_rejectThroughSample = 0;
};
