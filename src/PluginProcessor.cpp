#include "PluginProcessor.h"
#include "PluginEditor.h"

BassEmulatorProcessor::BassEmulatorProcessor()
    : AudioProcessor(BusesProperties()
        .withInput ("Input",  juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{}

BassEmulatorProcessor::~BassEmulatorProcessor() {}

juce::AudioProcessorValueTreeState::ParameterLayout BassEmulatorProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>("filterCutoff",    "Filter Cutoff",    100.0f, 2000.0f, 800.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("filterResonance", "Filter Resonance", 0.0f,   1.0f,    0.3f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("envAttack",       "Env Attack",       1.0f,   50.0f,   10.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("envRelease",      "Env Release",      10.0f,  500.0f,  100.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("dryWet",          "Dry/Wet",          0.0f,   1.0f,    1.0f));

    return { params.begin(), params.end() };
}

void BassEmulatorProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::dsp::ProcessSpec spec;
    spec.sampleRate       = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels      = 1;

    // Sawtooth: phase in [-pi, pi] mapped linearly to [-1, 1]
    osc.initialise([](float x) { return x / juce::MathConstants<float>::pi; }, 128);
    osc.prepare(spec);
    osc.setFrequency(currentPitch);

    filter.prepare(spec);
    filter.setMode(juce::dsp::LadderFilterMode::LPF12);

    onset.prepare(sampleRate);
    envFollower.prepare(sampleRate);
    yin.reset();

    bassBuffer.setSize(1, samplesPerBlock);
    pitchIsValid = false;
}

void BassEmulatorProcessor::releaseResources() {}

void BassEmulatorProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int    numSamples = buffer.getNumSamples();
    const float* inputData  = buffer.getReadPointer(0);

    // --- Analysis on input signal ---

    if (onset.process(inputData, numSamples))
        envFollower.triggerAttack();

    float detectedPitch = yin.process(inputData, numSamples, getSampleRate());
    if (detectedPitch > 0.0f)
    {
        currentPitch = detectedPitch / 2.0f;
        osc.setFrequency(currentPitch);
        pitchIsValid = true;
    }

    // Pass dry signal through until we have at least one valid pitch
    if (!pitchIsValid)
        return;

    // --- Synthesis ---

    filter.setCutoffFrequencyHz(apvts.getRawParameterValue("filterCutoff")->load());
    filter.setResonance(apvts.getRawParameterValue("filterResonance")->load());
    envFollower.setAttack(apvts.getRawParameterValue("envAttack")->load());
    envFollower.setRelease(apvts.getRawParameterValue("envRelease")->load());

    // Generate sawtooth into bassBuffer
    bassBuffer.clear();
    float* bassChannels[] = { bassBuffer.getWritePointer(0) };
    juce::dsp::AudioBlock<float>             bassBlock(bassChannels, 1, (size_t)numSamples);
    juce::dsp::ProcessContextReplacing<float> oscCtx(bassBlock);
    osc.process(oscCtx);

    // Shape amplitude with envelope (driven by input signal level)
    auto* bassData = bassBuffer.getWritePointer(0);
    for (int i = 0; i < numSamples; ++i)
        bassData[i] *= envFollower.processSample(inputData[i]);

    // Apply resonant lowpass filter
    juce::dsp::ProcessContextReplacing<float> filterCtx(bassBlock);
    filter.process(filterCtx);

    // Mix dry/wet into all output channels
    const float wet = apvts.getRawParameterValue("dryWet")->load();
    const float dry = 1.0f - wet;

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        auto* out = buffer.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
            out[i] = out[i] * dry + bassData[i] * wet;
    }
}

const juce::String BassEmulatorProcessor::getName() const        { return JucePlugin_Name; }
bool BassEmulatorProcessor::acceptsMidi() const                  { return false; }
bool BassEmulatorProcessor::producesMidi() const                 { return false; }
bool BassEmulatorProcessor::isMidiEffect() const                 { return false; }
double BassEmulatorProcessor::getTailLengthSeconds() const       { return 0.0; }
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