class Pcm16ChunkProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options.processorOptions || {};
    this.targetSampleRate = processorOptions.targetSampleRate || 16000;
    this.chunkSamples = processorOptions.chunkSamples || 1600;
    this.inputSamples = [];
    this.outputSamples = [];
    this.position = 0;
    this.ratio = sampleRate / this.targetSampleRate;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const frameCount = input[0].length;
    for (let index = 0; index < frameCount; index += 1) {
      let mixedSample = 0;
      for (let channelIndex = 0; channelIndex < input.length; channelIndex += 1) {
        mixedSample += input[channelIndex][index] || 0;
      }
      this.inputSamples.push(mixedSample / input.length);
    }

    while (this.position + 1 < this.inputSamples.length) {
      const leftIndex = Math.floor(this.position);
      const fraction = this.position - leftIndex;
      const left = this.inputSamples[leftIndex];
      const right = this.inputSamples[leftIndex + 1];
      this.outputSamples.push(left + (right - left) * fraction);
      this.position += this.ratio;

      if (this.outputSamples.length === this.chunkSamples) {
        const pcmBuffer = new ArrayBuffer(this.chunkSamples * 2);
        const pcmView = new DataView(pcmBuffer);
        for (let index = 0; index < this.chunkSamples; index += 1) {
          const sample = Math.max(-1, Math.min(1, this.outputSamples[index]));
          const integerSample = Math.round(sample < 0 ? sample * 0x8000 : sample * 0x7fff);
          pcmView.setInt16(index * 2, integerSample, true);
        }
        this.port.postMessage(pcmBuffer, [pcmBuffer]);
        this.outputSamples = [];
      }
    }

    // Keep the final source sample so the next render quantum can interpolate
    // across the boundary without drift or a discontinuity.
    const consumed = Math.min(
      Math.floor(this.position),
      Math.max(0, this.inputSamples.length - 1),
    );
    if (consumed > 0) {
      this.inputSamples.splice(0, consumed);
      this.position -= consumed;
    }
    return true;
  }
}

registerProcessor("pcm16-chunk-processor", Pcm16ChunkProcessor);
