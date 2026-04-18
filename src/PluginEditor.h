#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

class BassEmulatorEditor : public juce::AudioProcessorEditor
{
public:
    explicit BassEmulatorEditor(BassEmulatorProcessor&);
    ~BassEmulatorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    BassEmulatorProcessor& processor;

    juce::Slider cutoffSlider, resonanceSlider, attackSlider, releaseSlider, dryWetSlider;
    juce::Label  cutoffLabel,  resonanceLabel,  attackLabel,  releaseLabel,  dryWetLabel;

    juce::AudioProcessorValueTreeState::SliderAttachment
        cutoffAttachment, resonanceAttachment, attackAttachment, releaseAttachment, dryWetAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BassEmulatorEditor)
};