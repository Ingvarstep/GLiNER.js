import ort from "onnxruntime-web";
import { ONNXWrapper } from "./ONNXWrapper";

export class Model {
  constructor(
    private config: any,
    private processor: any,
    private decoder: any,
    private onnxWrapper: ONNXWrapper
  ) { }

  async initialize(): Promise<void> {
    await this.onnxWrapper.init()
  }

  prepareInputs(batch: any): Record<string, ort.Tensor> {
    const batch_size = batch.inputsIds.length;
    const num_tokens = batch.inputsIds[0].length;
    const num_spans = batch.spanIdxs[0].length;

    const convertToBool = (arr: any[]): any[] => {
      return arr.map((subArr) => subArr.map((val: any) => !!val));
    };

    const blankConvert = (arr: any[]): any[] => {
      return arr.map((val) => (isNaN(Number(val)) ? 0 : Math.floor(Number(val))));
    };

    const convertToInt = (arr: any[]): any[] => {
      return arr.map((subArr) =>
        subArr.map((val: any) => (isNaN(Number(val)) ? 0 : Math.floor(Number(val))))
      );
    };

    const convertSpanIdx = (arr: any[]): any[] => {
      return arr.flatMap((subArr) =>
        subArr.flatMap((pair: any) => pair.map((val: any) => (isNaN(Number(val)) ? 0 : Math.floor(Number(val)))))
      );
    };

    const createTensor = (
      data: any[],
      shape: number[],
      conversionFunc: (arr: any[]) => any[] = convertToInt,
      tensorType: any = "int64"
    ): ort.Tensor => {
      const convertedData = conversionFunc(data);
      return new this.onnxWrapper.ort.Tensor(tensorType, convertedData.flat(Infinity), shape);
    };

    let input_ids = createTensor(batch.inputsIds, [batch_size, num_tokens]);
    let attention_mask = createTensor(batch.attentionMasks, [batch_size, num_tokens], convertToBool); // NOTE: why convert to bool but type is not bool?
    let words_mask = createTensor(batch.wordsMasks, [batch_size, num_tokens]);
    let text_lengths = createTensor(batch.textLengths, [batch_size, 1], blankConvert);
    let span_idx = createTensor(batch.spanIdxs, [batch_size, num_spans, 2], convertSpanIdx);
    let span_mask = createTensor(batch.spanMasks, [batch_size, num_spans], convertToBool, "bool");

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
    multiLabel: boolean = false
  ): Promise<number[][][]> {
    let batch = this.processor.prepareBatch(texts, entities);
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
      batch.batchWordsStartIdx,
      batch.batchWordsEndIdx,
      batch.idToClass,
      modelOutput,
      flatNer,
      threshold,
      multiLabel
    );

    return decodedSpans;
  }
}
