#include "PluginEditor.h"

BassEmulatorEditor::BassEmulatorEditor(BassEmulatorProcessor& p)
    : AudioProcessorEditor(&p), processor(p),
      roomSizeAttachment(p.apvts, "roomSize", roomSizeSlider),
      dampingAttachment (p.apvts, "damping",  dampingSlider),
      wetAttachment     (p.apvts, "wetLevel", wetSlider),
      dryAttachment     (p.apvts, "dryLevel", drySlider)
{
    auto setupSlider = [this](juce::Slider& s) {
        s.setSliderStyle(juce::Slider::RotaryVerticalDrag);
        s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
        addAndMakeVisible(s);
    };
    setupSlider(roomSizeSlider);
    setupSlider(dampingSlider);
    setupSlider(wetSlider);
    setupSlider(drySlider);

    auto setupLabel = [this](juce::Label& l, const juce::String& text) {
        l.setText(text, juce::dontSendNotification);
        l.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(l);
    };
    setupLabel(roomSizeLabel, "Room");
    setupLabel(dampingLabel,  "Damping");
    setupLabel(wetLabel,      "Wet");
    setupLabel(dryLabel,      "Dry");

    setSize(400, 200);
}

BassEmulatorEditor::~BassEmulatorEditor() {}

void BassEmulatorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::darkgrey);
}

void BassEmulatorEditor::resized()
{
    auto area = getLocalBounds().reduced(10);
    int sliderWidth = area.getWidth() / 4;
    int labelHeight = 20;

    juce::Label*  labels[]  = { &roomSizeLabel, &dampingLabel, &wetLabel, &dryLabel };
    juce::Slider* sliders[] = { &roomSizeSlider, &dampingSlider, &wetSlider, &drySlider };

    for (int i = 0; i < 4; ++i)
    {
        auto col = area.removeFromLeft(sliderWidth);
        labels[i]->setBounds(col.removeFromTop(labelHeight));
        sliders[i]->setBounds(col);
    }
}