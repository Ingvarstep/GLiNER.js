import ort from "onnxruntime-web";
import { ONNXWrapper } from "./ONNXWrapper";
import { RawInferenceResult } from "./Gliner";

export class Model {
  constructor(
    private config: any,
    private processor: any,
    private decoder: any,
    private onnxWrapper: ONNXWrapper,
  ) { }

  async initialize(): Promise<void> {
    await this.onnxWrapper.init();
  }

  prepareInputs(batch: any): Record<string, ort.Tensor> {
    const batch_size = batch.inputsIds.length;
    const num_tokens = batch.inputsIds[0].length;
    const num_spans = batch.spanIdxs[0].length;

    const createTensor = (
      data: any[],
      shape: number[],
      tensorType: any = "int64",
    ): ort.Tensor => {
      // @ts-ignore // NOTE: node types not working
      return new this.onnxWrapper.ort.Tensor(
        tensorType,
        data.flat(Infinity),
        shape,
      );
    };

    let input_ids = createTensor(batch.inputsIds, [batch_size, num_tokens]);
    let attention_mask = createTensor(batch.attentionMasks, [
      batch_size,
      num_tokens,
    ]); // NOTE: why convert to bool but type is not bool?
    let words_mask = createTensor(batch.wordsMasks, [batch_size, num_tokens]);
    let text_lengths = createTensor(batch.textLengths, [batch_size, 1]);
    let span_idx = createTensor(batch.spanIdxs, [batch_size, num_spans, 2]);
    let span_mask = createTensor(
      batch.spanMasks,
      [batch_size, num_spans],
      "bool",
    );

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
    texts: string[],
    entities: string[],
    flatNer: boolean = false,
    threshold: number = 0.5,
    multiLabel: boolean = false,
  ): Promise<RawInferenceResult> {
    let batch = this.processor.prepareBatch(texts, entities);
    let feeds = this.prepareInputs(batch);
    const results = await this.onnxWrapper.run(feeds);
    const modelOutput = results.logits.data;
    // const modelOutput = results.logits.data as number[];

    const batchSize = batch.batchTokens.length;
    const inputLength = Math.max(...batch.textLengths);
    const maxWidth = this.config.max_width;
    const numEntities = entities.length;
    const batchIds = Array.from({ length: batchSize }, (_, i) => i);
    const decodedSpans = this.decoder.decode(
      batchSize,
      inputLength,
      maxWidth,
      numEntities,
      texts,
      batchIds,
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

  async inference_with_chunking(
    texts: string[],
    entities: string[],
    flatNer: boolean = false,
    threshold: number = 0.5,
    multiLabel: boolean = false,
    batch_size: number = 4,
    max_words: number = 512,
  ): Promise<RawInferenceResult> {
    const { classToId, idToClass } = this.processor.createMappings(entities);

    let batchIds: number[] = [];
    let batchTokens: number[][] = [];
    let batchWordsStartIdx: number[][] = [];
    let batchWordsEndIdx: number[][] = [];
    texts.forEach((text, id) => {
      let [tokens, wordsStartIdx, wordsEndIdx] =
        this.processor.tokenizeText(text);
      let num_sub_batches: number = Math.ceil(tokens.length / max_words);

      for (let i = 0; i < num_sub_batches; i++) {
        let start = i * max_words;
        let end = Math.min((i + 1) * max_words, tokens.length);

        batchIds.push(id);
        batchTokens.push(tokens.slice(start, end));
        batchWordsStartIdx.push(wordsStartIdx.slice(start, end));
        batchWordsEndIdx.push(wordsEndIdx.slice(start, end));
      }
    });

    let num_batches: number = Math.ceil(batchIds.length / batch_size);

    let finalDecodedSpans: RawInferenceResult = [];
    for (let id = 0; id < texts.length; id++) {
      finalDecodedSpans.push([]);
    }

    for (let batch_id = 0; batch_id < num_batches; batch_id++) {
      let start: number = batch_id * batch_size;
      let end: number = Math.min((batch_id + 1) * batch_size, batchIds.length);

      let currBatchTokens = batchTokens.slice(start, end);
      let currBatchWordsStartIdx = batchWordsStartIdx.slice(start, end);
      let currBatchWordsEndIdx = batchWordsEndIdx.slice(start, end);
      let currBatchIds = batchIds.slice(start, end);

      let [inputTokens, textLengths, promptLengths] =
        this.processor.prepareTextInputs(currBatchTokens, entities);

      let [inputsIds, attentionMasks, wordsMasks] = this.processor.encodeInputs(
        inputTokens,
        promptLengths,
      );

      inputsIds = this.processor.padArray(inputsIds);
      attentionMasks = this.processor.padArray(attentionMasks);
      wordsMasks = this.processor.padArray(wordsMasks);

      let { spanIdxs, spanMasks } = this.processor.prepareSpans(
        currBatchTokens,
        this.config["max_width"],
      );

      spanIdxs = this.processor.padArray(spanIdxs, 3);
      spanMasks = this.processor.padArray(spanMasks);

      let batch = {
        inputsIds: inputsIds,
        attentionMasks: attentionMasks,
        wordsMasks: wordsMasks,
        textLengths: textLengths,
        spanIdxs: spanIdxs,
        spanMasks: spanMasks,
        idToClass: idToClass,
        batchTokens: currBatchTokens,
        batchWordsStartIdx: currBatchWordsStartIdx,
        batchWordsEndIdx: currBatchWordsEndIdx,
      };

      let feeds = this.prepareInputs(batch);
      const results = await this.onnxWrapper.run(feeds);
      const modelOutput = results.logits.data;
      // const modelOutput = results.logits.data as number[];

      const batchSize = batch.batchTokens.length;
      const inputLength = Math.max(...batch.textLengths);
      const maxWidth = this.config.max_width;
      const numEntities = entities.length;

      const decodedSpans = this.decoder.decode(
        batchSize,
        inputLength,
        maxWidth,
        numEntities,
        texts,
        currBatchIds,
        batch.batchWordsStartIdx,
        batch.batchWordsEndIdx,
        batch.idToClass,
        modelOutput,
        flatNer,
        threshold,
        multiLabel,
      );

      for (let i = 0; i < currBatchIds.length; i++) {
        const originalTextId = currBatchIds[i];
        finalDecodedSpans[originalTextId].push(...decodedSpans[i]);
      }
    }

    return finalDecodedSpans;
  }
}
