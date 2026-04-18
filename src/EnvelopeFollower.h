#pragma once
#include <cmath>

// Simple RC-chain envelope follower with independent attack and release times.
class EnvelopeFollower
{
public:
    EnvelopeFollower() = default;

    // Call from prepareToPlay
    void prepare(double sr, float attackMs = 10.0f, float releaseMs = 100.0f)
    {
        sampleRate = sr;
        setAttack(attackMs);
        setRelease(releaseMs);
        reset();
    }

    void setAttack(float ms)
    {
        attackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * ms / 1000.0f));
    }

    void setRelease(float ms)
    {
        releaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * ms / 1000.0f));
    }

    void reset() { envelope = 0.0f; }

    // Call on onset to snap envelope to zero for a clean attack transient
    void triggerAttack() { envelope = 0.0f; }

    // Process one sample; returns current envelope value (0..1)
    inline float processSample(float sample)
    {
        float level = std::abs(sample);
        float coeff = (level > envelope) ? attackCoeff : releaseCoeff;
        envelope    = coeff * envelope + (1.0f - coeff) * level;
        return envelope;
    }

    // Process a block and return the last envelope value
    float processBlock(const float* samples, int numSamples)
    {
        for (int i = 0; i < numSamples; ++i)
            processSample(samples[i]);
        return envelope;
    }

private:
    double sampleRate   = 48000.0;
    float  attackCoeff  = 0.0f;
    float  releaseCoeff = 0.0f;
    float  envelope     = 0.0f;
};