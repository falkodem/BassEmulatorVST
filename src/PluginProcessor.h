#pragma once
#include <JuceHeader.h>
#include "YinPitchDetector.h"
#include "OnsetDetector.h"
#include "EnvelopeFollower.h"

class BassEmulatorProcessor : public juce::AudioProcessor
{
public:
    BassEmulatorProcessor();
    ~BassEmulatorProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

private:
    juce::dsp::Oscillator<float>   osc;
    juce::dsp::LadderFilter<float> filter;

    YinPitchDetector yin;
    OnsetDetector    onset;
    EnvelopeFollower envFollower;

    juce::AudioBuffer<float> bassBuffer;
    float currentPitch = 110.0f;
    bool  pitchIsValid = false;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BassEmulatorProcessor)
};