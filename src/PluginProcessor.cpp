#include "PluginProcessor.h"
#include "PluginEditor.h"

BassEmulatorProcessor::BassEmulatorProcessor()
    : AudioProcessor(BusesProperties()
        .withInput ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
}

BassEmulatorProcessor::~BassEmulatorProcessor() {}

juce::AudioProcessorValueTreeState::ParameterLayout BassEmulatorProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>("roomSize", "Room Size", 0.0f, 1.0f, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("damping",  "Damping",   0.0f, 1.0f, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("wetLevel", "Wet Level", 0.0f, 1.0f, 0.33f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("dryLevel", "Dry Level", 0.0f, 1.0f, 0.4f));

    return { params.begin(), params.end() };
}

void BassEmulatorProcessor::updateReverbParameters()
{
    reverbParams.roomSize  = apvts.getRawParameterValue("roomSize")->load();
    reverbParams.damping   = apvts.getRawParameterValue("damping")->load();
    reverbParams.wetLevel  = apvts.getRawParameterValue("wetLevel")->load();
    reverbParams.dryLevel  = apvts.getRawParameterValue("dryLevel")->load();
    reverb.setParameters(reverbParams);
}

void BassEmulatorProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    reverb.setSampleRate(sampleRate);
    updateReverbParameters();
}

void BassEmulatorProcessor::releaseResources() {}

void BassEmulatorProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    updateReverbParameters();

    if (getTotalNumInputChannels() == 1)
        reverb.processMono(buffer.getWritePointer(0), buffer.getNumSamples());
    else
        reverb.processStereo(buffer.getWritePointer(0), buffer.getWritePointer(1), buffer.getNumSamples());
}

const juce::String BassEmulatorProcessor::getName() const        { return JucePlugin_Name; }
bool BassEmulatorProcessor::acceptsMidi() const                  { return false; }
bool BassEmulatorProcessor::producesMidi() const                 { return false; }
bool BassEmulatorProcessor::isMidiEffect() const                 { return false; }
double BassEmulatorProcessor::getTailLengthSeconds() const       { return 2.0; }
int BassEmulatorProcessor::getNumPrograms()                      { return 1; }
int BassEmulatorProcessor::getCurrentProgram()                   { return 0; }
void BassEmulatorProcessor::setCurrentProgram(int)               {}
const juce::String BassEmulatorProcessor::getProgramName(int)    { return {}; }
void BassEmulatorProcessor::changeProgramName(int, const juce::String&) {}
bool BassEmulatorProcessor::hasEditor() const                    { return true; }

void BassEmulatorProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void BassEmulatorProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState && xmlState->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
}

juce::AudioProcessorEditor* BassEmulatorProcessor::createEditor()
{
    return new BassEmulatorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new BassEmulatorProcessor();
}