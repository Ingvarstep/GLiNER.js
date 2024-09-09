// const ort = require('onnxruntime-web');
import ort from "onnxruntime-node";

export class Model {
  constructor(config, processor, decoder, modelPath = "./model.onnx") {
    this.config = config;
    this.processor = processor;
    this.decoder = decoder;
    this.modelPath = modelPath;
    this.session = null;
  }

  async initialize() {
    if (!this.session) {
      this.session = await ort.InferenceSession.create(this.modelPath);
    }
  }

  prepareInputs(batch) {
    const batch_size = batch.inputsIds.length;
    const num_tokens = batch.inputsIds[0].length;
    const num_spans = batch.spanIdxs[0].length;

    // Conversion function to boolean (for masks)
    const convertToBool = (arr) => {
      return arr.map(
        (subArr) => subArr.map((val) => !!val), // Convert each value to boolean
      );
    };

    const blankConvert = (arr) => {
      return arr.map((val) =>
        isNaN(Number(val)) ? 0 : Math.floor(Number(val)),
      );
    };

    const convertToInt = (arr) => {
      return arr.map((subArr) =>
        subArr.map((val) => (isNaN(Number(val)) ? 0 : Math.floor(Number(val)))),
      );
    };

    // Special conversion for span_idx
    const convertSpanIdx = (arr) => {
      return arr.flatMap((subArr) =>
        subArr.flatMap((pair) =>
          pair.map((val) => (isNaN(Number(val)) ? 0 : Math.floor(Number(val)))),
        ),
      );
    };

    // Helper function to create tensors safely, with support for different tensor types
    const createTensor = (
      data,
      shape,
      conversionFunc = convertToInt,
      tensorType = "int64",
    ) => {
      const convertedData = conversionFunc(data);
      return new ort.Tensor(tensorType, convertedData.flat(Infinity), shape);
    };

    // Prepare tensors for ONNX input
    let input_ids = createTensor(batch.inputsIds, [batch_size, num_tokens]);
    let attention_mask = createTensor(
      batch.attentionMasks,
      [batch_size, num_tokens],
      convertToBool,
    );
    let words_mask = createTensor(
      batch.wordsMasks,
      [batch_size, num_tokens],
      convertToBool,
    );
    let text_lengths = createTensor(
      batch.textLengths,
      [batch_size, 1],
      blankConvert,
    );
    let span_idx = createTensor(
      batch.spanIdxs,
      [batch_size, num_spans, 2],
      convertSpanIdx,
    );
    let span_mask = createTensor(
      batch.spanMasks,
      [batch_size, num_spans],
      convertToBool,
      "bool",
    );

    // Create feeds for ONNX inference
    const feeds = {
      input_ids: input_ids,
      attention_mask: attention_mask,
      words_mask: words_mask,
      text_lengths: text_lengths,
      span_idx: span_idx,
      span_mask: span_mask,
    };

    return feeds;
  }

  async inference(
    texts,
    entities,
    flatNer = false,
    threshold = 0.5,
    multiLabel = false,
  ) {
    let batch = this.processor.prepareBatch(texts, entities);

    let feeds = this.prepareInputs(batch);

    const results = await this.session.run(feeds);
    
    const modelOutput = results.logits.data;

    const batchSize = batch.batchTokens.length;
    const inputLength = Math.max(...batch.textLengths);
    const maxWidth = this.config.max_width;
    const numEntities = entities.length;

    const decodedSpans = this.decoder.decode(
      batchSize,
      inputLength,
      maxWidth,
      numEntities,
      batch.batchWordsStartIdx,
      batch.batchWordsEndIdx,
      batch.idToClass,
      modelOutput,
      flatNer,
      threshold,
      multiLabel,
    );

    return decodedSpans;
  }
}
