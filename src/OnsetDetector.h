#pragma once
#include <cmath>

// Energy-based onset detector.
// Returns true when a sudden amplitude rise is detected (new note attack).
class OnsetDetector
{
public:
    OnsetDetector() = default;

    // Call from prepareToPlay
    void prepare(double sampleRate, float sensitivityDb = 6.0f, float cooldownMs = 50.0f)
    {
        // sensitivityDb: how much louder the new frame must be to count as onset
        energyThreshold  = std::pow(10.0f, sensitivityDb / 20.0f);
        cooldownDuration = static_cast<int>(sampleRate * cooldownMs / 1000.0);
        reset();
    }

    void reset()
    {
        prevEnergy      = 0.0f;
        cooldownCounter = 0;
    }

    // Feed one processBlock's worth of mono samples.
    // Returns true on the block where an onset is detected.
    bool process(const float* samples, int numSamples)
    {
        float energy = computeRMS(samples, numSamples);

        bool onset = false;

        if (cooldownCounter > 0)
        {
            cooldownCounter -= numSamples;
            if (cooldownCounter < 0) cooldownCounter = 0;
        }
        else if (prevEnergy > kMinEnergy && energy / prevEnergy > energyThreshold)
        {
            onset           = true;
            cooldownCounter = cooldownDuration;
        }

        prevEnergy = energy;
        return onset;
    }

private:
    // Below this level we treat signal as silence and skip detection
    static constexpr float kMinEnergy = 1e-4f;

    float energyThreshold  = 2.0f;   // ratio (linear), set in prepare()
    int   cooldownDuration = 2400;   // samples, set in prepare()
    float prevEnergy       = 0.0f;
    int   cooldownCounter  = 0;

    float computeRMS(const float* samples, int numSamples) const
    {
        float sum = 0.0f;
        for (int i = 0; i < numSamples; ++i)
            sum += samples[i] * samples[i];
        return std::sqrt(sum / static_cast<float>(numSamples));
    }
};