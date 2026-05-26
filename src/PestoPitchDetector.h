#pragma once

#include <atomic>
#include <vector>
#include <anira/anira.h>
#include <PestoModelData.h>

/**
 * PESTO pitch detector обёртка поверх ANIRA + ONNX Runtime.
 *
 * Модель `models/pesto.onnx` встроена через `juce_add_binary_data` (см. CMakeLists.txt)
 * и доступна как `PestoModelData::pesto_onnx`. Параметры окна/хопа должны совпадать
 * с экспортом `ml/utils/export_pesto_onnx.py` — мы фиксируем sr=44100 и hop=10 мс.
 *
 * Архитектура:
 *   - Аудио идёт стримом через ANIRA RingBuffer (preprocess_input_size = hop).
 *   - PESTO выход — НЕ аудио, а два вектора (F0, confidence). Помечаем их как
 *     non-streamable (postprocess_output_size = 0): ANIRA кладёт их в потокобезопасное
 *     atomic-хранилище, читаем оттуда из audio-треда без аллокаций и блокировок.
 *   - Кастомный pre_process берёт перекрывающиеся окна (новые kHopSamples сэмплов +
 *     kWindowSamples - kHopSamples от прошлого вызова), post_process — default.
 *
 * API мимикрирует YinPitchDetector: `process(buf, n)` возвращает F0 (Гц) или 0.0f
 * пока валидного питча ещё нет (initial buffering или unvoiced).
 */
class PestoPitchDetector
{
public:
    // ── параметры модели (должны совпадать с pesto.onnx) ──────────────────────
    static constexpr int          kSampleRate      = 44100;
    static constexpr int          kHopSamples      = 441;     // 10 мс
    static constexpr int          kWindowSamples   = 11025;   // ~250 мс — контекст последнего кадра
    static constexpr int          kFramesPerWindow = 26;      // kWindowSamples/kHopSamples + 1, см. meta.json
    static constexpr float        kVoicedThreshold = 0.5f;    // PESTO confidence порог
    static constexpr float        kMaxInferenceMs  = 20.0f;   // бюджет на одну инференс-вызов (замер: ~5–6 мс)
    static constexpr unsigned int kWarmUp          = 2;       // прогрев чтобы первый реальный инференс не тормозил

    PestoPitchDetector()
        : m_config(
              { anira::ModelData(
                    reinterpret_cast<void*>(const_cast<char*>(PestoModelData::pesto_onnx)),
                    static_cast<size_t>(PestoModelData::pesto_onnxSize),
                    anira::InferenceBackend::ONNX) },
              { anira::TensorShape({ { kWindowSamples } },
                                   { { kFramesPerWindow }, { kFramesPerWindow } },
                                   anira::InferenceBackend::ONNX) },
              anira::ProcessingSpec(
                  /* preprocess_input_channels   */ { 1 },
                  /* postprocess_output_channels */ { 1, 1 },
                  /* preprocess_input_size       */ { static_cast<size_t>(kHopSamples) },
                  /* postprocess_output_size     */ { 0, 0 }   // 0 -> non-streamable atomic storage
              ),
              kMaxInferenceMs,
              kWarmUp),
          m_processor(m_config),
          m_handler(m_processor, m_config)
    {}

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
        m_hasValid.store(false, std::memory_order_release);
        m_lastPitch.store(0.0f, std::memory_order_relaxed);
    }

    /** Толкаем блок аудио в инференс-пайплайн и возвращаем последний валидный F0 (Гц).
     *  Возврат 0.0f означает «питч ещё не определён» — синтез должен пропустить блок. */
    float process(const float* monoInput, int numSamples)
    {
        const float* const ptrs[1] = { monoInput };
        m_handler.push_data(ptrs, static_cast<size_t>(numSamples));

        // Берём кадр в самой свежей позиции окна. У него меньше «будущего контекста»
        // (zero-pad), но он соответствует «сейчас». Можно сдвинуться на -7 кадров
        // для лучшего контекста ценой +70 мс задержки — оставлено на будущее (см. Block 5).
        const float conf = m_processor.get_output(1, kFramesPerWindow - 1);
        if (conf >= kVoicedThreshold)
        {
            const float f0 = m_processor.get_output(0, kFramesPerWindow - 1);
            if (f0 > 0.0f)
            {
                m_lastPitch.store(f0, std::memory_order_relaxed);
                m_hasValid.store(true, std::memory_order_release);
            }
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

    /** Латентность инференс-конвейера в сэмплах (для setLatencySamples в processor). */
    int getLatencySamples() const
    {
        return static_cast<int>(m_handler.get_latency());
    }

private:
    // Перекрывающиеся окна: на каждый триггер инференса (раз в kHopSamples сэмплов)
    // вынимаем kHopSamples новых + (kWindowSamples - kHopSamples) старых из ring-buffer.
    // post_process не переопределяем — default impl кладёт non-streamable выходы
    // в atomic storage, читаемое через m_processor.get_output().
    struct PestoProcessor : anira::PrePostProcessor
    {
        using anira::PrePostProcessor::PrePostProcessor;
        void pre_process(std::vector<anira::RingBuffer>& input,
                         std::vector<anira::BufferF>& output,
                         anira::InferenceBackend /*backend*/) override
        {
            pop_samples_from_buffer(input[0], output[0],
                                    static_cast<size_t>(PestoPitchDetector::kHopSamples),
                                    static_cast<size_t>(PestoPitchDetector::kWindowSamples
                                                        - PestoPitchDetector::kHopSamples));
        }
    };

    anira::InferenceConfig  m_config;
    PestoProcessor          m_processor;
    anira::InferenceHandler m_handler;

    std::atomic<float> m_lastPitch { 0.0f };
    std::atomic<bool>  m_hasValid  { false };
};
