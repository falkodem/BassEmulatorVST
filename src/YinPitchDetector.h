#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

// YIN pitch detection algorithm (de Cheveigne & Kawahara, 2002)
// Processes mono audio, returns fundamental frequency in Hz or -1 if undetected.
class YinPitchDetector
{
public:
    YinPitchDetector(int windowSize = 1024, float threshold = 0.15f)
        : windowSize(windowSize)
        , halfWindow(windowSize / 2)
        , bufSize(windowSize + windowSize / 2)
        , threshold(threshold)
        , inputBuffer(windowSize + windowSize / 2, 0.0f)
        , yinBuffer(windowSize / 2, 0.0f)
    {}

    void reset()
    {
        std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
        writePos            = 0;
        totalSamples        = 0;
        samplesSinceLastRun = 0;
        lastPitch           = -1.0f;
    }

    // Feed samples from processBlock; returns Hz or -1 if undetected.
    // Pitch is updated once per windowSize samples (~21 ms at 48 kHz, window=1024).
    float process(const float* samples, int numSamples, double sampleRate)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            inputBuffer[writePos] = samples[i];
            writePos = (writePos + 1) % bufSize;
            ++totalSamples;
            ++samplesSinceLastRun;
        }

        if (totalSamples < bufSize)
            return -1.0f;

        if (samplesSinceLastRun >= windowSize)
        {
            samplesSinceLastRun = 0;
            lastPitch = runYin(sampleRate);
        }

        return lastPitch;
    }

private:
    int   windowSize;
    int   halfWindow;
    int   bufSize;
    float threshold;

    std::vector<float> inputBuffer;
    std::vector<float> yinBuffer;

    int   writePos            = 0;
    int   totalSamples        = 0;
    int   samplesSinceLastRun = 0;
    float lastPitch           = -1.0f;

    // Ring buffer read: index 0 = oldest sample
    inline float getSample(int index) const
    {
        return inputBuffer[(writePos + index) % bufSize];
    }

    float runYin(double sampleRate)
    {
        differenceFunction();
        cumulativeMeanNormalizedDifference();

        int tau = absoluteThreshold();
        if (tau < 2)
            return -1.0f;

        float refined = parabolicInterpolation(tau);
        float freq    = static_cast<float>(sampleRate / refined);

        // Guitar range: E2 (82 Hz) to roughly D6 (1175 Hz)
        if (freq < 70.0f || freq > 1300.0f)
            return -1.0f;

        return freq;
    }

    // Step 1: squared difference between signal and its shifted copy
    void differenceFunction()
    {
        for (int tau = 0; tau < halfWindow; ++tau)
        {
            float sum = 0.0f;
            for (int j = 0; j < windowSize; ++j)
            {
                float diff = getSample(j) - getSample(j + tau);
                sum += diff * diff;
            }
            yinBuffer[tau] = sum;
        }
    }

    // Step 2: normalize so tau=0 is always 1, removing bias toward small lags
    void cumulativeMeanNormalizedDifference()
    {
        yinBuffer[0]    = 1.0f;
        float runningSum = 0.0f;
        for (int tau = 1; tau < halfWindow; ++tau)
        {
            runningSum     += yinBuffer[tau];
            yinBuffer[tau]  = (runningSum > 0.0f)
                ? yinBuffer[tau] * static_cast<float>(tau) / runningSum
                : 0.0f;
        }
    }

    // Step 3: first lag below threshold that is a local minimum
    int absoluteThreshold()
    {
        for (int tau = 2; tau < halfWindow; ++tau)
        {
            if (yinBuffer[tau] < threshold)
            {
                while (tau + 1 < halfWindow && yinBuffer[tau + 1] < yinBuffer[tau])
                    ++tau;
                return tau;
            }
        }
        // Fallback: global minimum (low-confidence result)
        return static_cast<int>(
            std::min_element(yinBuffer.begin() + 2, yinBuffer.end()) - yinBuffer.begin()
        );
    }

    // Step 4: sub-sample refinement via parabolic interpolation
    float parabolicInterpolation(int tau)
    {
        if (tau <= 0 || tau >= halfWindow - 1)
            return static_cast<float>(tau);

        float s0    = yinBuffer[tau - 1];
        float s1    = yinBuffer[tau];
        float s2    = yinBuffer[tau + 1];
        float denom = 2.0f * (2.0f * s1 - s0 - s2);

        if (std::abs(denom) < 1e-7f)
            return static_cast<float>(tau);

        return static_cast<float>(tau) + (s2 - s0) / denom;
    }
};
