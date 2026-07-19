#include "PluginEditor.h"

BassEmulatorEditor::BassEmulatorEditor(BassEmulatorProcessor& p)
    : AudioProcessorEditor(&p), processor(p),
      cutoffAttachment    (p.apvts, "filterCutoff",    cutoffSlider),
      resonanceAttachment (p.apvts, "filterResonance", resonanceSlider),
      attackAttachment    (p.apvts, "envAttack",       attackSlider),
      releaseAttachment   (p.apvts, "envRelease",      releaseSlider),
      dryWetAttachment    (p.apvts, "dryWet",          dryWetSlider),
      waveformAttachment  (p.apvts, "waveform",        waveformBox)
{
    auto setupSlider = [this](juce::Slider& s) {
        s.setSliderStyle(juce::Slider::RotaryVerticalDrag);
        s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
        addAndMakeVisible(s);
    };
    setupSlider(cutoffSlider);
    setupSlider(resonanceSlider);
    setupSlider(attackSlider);
    setupSlider(releaseSlider);
    setupSlider(dryWetSlider);

    auto setupLabel = [this](juce::Label& l, const juce::String& text) {
        l.setText(text, juce::dontSendNotification);
        l.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(l);
    };
    setupLabel(cutoffLabel,    "Cutoff");
    setupLabel(resonanceLabel, "Resonance");
    setupLabel(attackLabel,    "Attack");
    setupLabel(releaseLabel,   "Release");
    setupLabel(dryWetLabel,    "Dry/Wet");
    setupLabel(waveformLabel,   "Waveform");

    waveformBox.addItem("Saw",  1);
    waveformBox.addItem("Sine", 2);
    waveformBox.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(waveformBox);

    setSize(600, 200);
}

BassEmulatorEditor::~BassEmulatorEditor() {}

void BassEmulatorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::darkgrey);
}

void BassEmulatorEditor::resized()
{
    auto area = getLocalBounds().reduced(10);
    const int controlWidth = area.getWidth() / 6;
    const int labelHeight = 20;

    juce::Label*  labels[]  = { &cutoffLabel, &resonanceLabel, &attackLabel, &releaseLabel, &dryWetLabel };
    juce::Slider* sliders[] = { &cutoffSlider, &resonanceSlider, &attackSlider, &releaseSlider, &dryWetSlider };

    for (int i = 0; i < 5; ++i)
    {
        auto col = area.removeFromLeft(controlWidth);
        labels[i]->setBounds(col.removeFromTop(labelHeight));
        sliders[i]->setBounds(col);
    }

    auto waveformCol = area.removeFromLeft(controlWidth);
    waveformLabel.setBounds(waveformCol.removeFromTop(labelHeight));
    waveformBox.setBounds(waveformCol.removeFromTop(28).reduced(5, 2));
}
