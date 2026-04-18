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

    juce::Slider roomSizeSlider, dampingSlider, wetSlider, drySlider;
    juce::Label  roomSizeLabel,  dampingLabel,  wetLabel,  dryLabel;

    juce::AudioProcessorValueTreeState::SliderAttachment
        roomSizeAttachment, dampingAttachment, wetAttachment, dryAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BassEmulatorEditor)
};