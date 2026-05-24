cmake_minimum_required(VERSION 3.22)
project(BassEmulatorVST VERSION 0.1.0)

# ANIRA требует C++20 (наш плагин до этого собирался на C++17).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ── JUCE ──────────────────────────────────────────────────────────────────────
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        7.0.12
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(JUCE)

# ── ANIRA + ONNX Runtime ──────────────────────────────────────────────────────
# Inference engine для PESTO (Phase 2A, Шаг 2). ANIRA сама подтягивает
# бинарники ONNX Runtime под целевую платформу. Бэкенды LibTorch / TFLite
# отключены — нужен только ORT. Тесты/бенчмарки/примеры/доки ANIRA по умолчанию
# включены — выключаем, чтобы не собирать лишнее.
set(ANIRA_WITH_ONNXRUNTIME ON  CACHE BOOL "" FORCE)
set(ANIRA_WITH_LIBTORCH    OFF CACHE BOOL "" FORCE)
set(ANIRA_WITH_TFLITE      OFF CACHE BOOL "" FORCE)
set(ANIRA_WITH_TESTS       OFF CACHE BOOL "" FORCE)
set(ANIRA_WITH_BENCHMARK   OFF CACHE BOOL "" FORCE)
set(ANIRA_WITH_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(ANIRA_WITH_DOCS        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    anira
    GIT_REPOSITORY https://github.com/anira-project/anira.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(anira)

# ── Плагин ────────────────────────────────────────────────────────────────────
juce_add_plugin(BassEmulatorVST
    PLUGIN_MANUFACTURER_CODE Dmbt
    PLUGIN_CODE BVR1
    FORMATS VST3
    PRODUCT_NAME "BassEmulatorVST"
    COMPANY_NAME "Personal"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    VST3_CATEGORIES "Fx"
)

juce_generate_juce_header(BassEmulatorVST)

target_sources(BassEmulatorVST PRIVATE
    src/PluginProcessor.cpp
    src/PluginEditor.cpp
)

target_compile_definitions(BassEmulatorVST PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(BassEmulatorVST
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        anira::anira
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# ── Копирование ORT shared libs рядом с .vst3 (Windows) ───────────────────────
# ANIRA выставляет ANIRA_SHARED_LIBS_WIN — список DLL бэкенда. Если их не
# положить в Contents/x86_64-win/ рядом с BassEmulatorVST.vst3 — хост (Reaper)
# не сможет загрузить плагин из-за неразрешённых импортов onnxruntime.dll.
if (MSVC)
    if (DEFINED ANIRA_SHARED_LIBS_WIN)
        add_custom_command(TARGET BassEmulatorVST_VST3 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ANIRA_SHARED_LIBS_WIN}
                $<TARGET_FILE_DIR:BassEmulatorVST_VST3>
        )
    else()
        message(WARNING "ANIRA_SHARED_LIBS_WIN не определён — ORT DLL не "
                        "будет скопирован к VST3. Плагин может не загрузиться.")
    endif()
endif()
