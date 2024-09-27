import { AutoTokenizer, env } from "@xenova/transformers";
import { Model } from "./model";
import { WhitespaceTokenSplitter, SpanProcessor } from "./processor";
import { SpanDecoder } from "./decoder";
import { IONNXSettings, ONNXWrapper } from "./ONNXWrapper";

export interface ITransformersSettings {
  allowLocalModels: boolean;
  useBrowserCache: boolean;
}

export interface InitConfig {
  tokenizerPath: string;
  onnxSettings: IONNXSettings;
  transformersSettings?: ITransformersSettings;
  maxWidth?: number;
}

export interface IInference {
  texts: string[];
  entities: string[];
  flatNer?: boolean;
  threshold?: number;
}

export type RawInferenceResult = [string, number, number, string, number][][];

export interface IEntityResult {
  spanText: string;
  start: number;
  end: number;
  label: string;
  score: number;
}
export type InferenceResultSingle = IEntityResult[];
export type InferenceResultMultiple = InferenceResultSingle[];

export class Gliner {
  private model: Model | null = null;

  constructor(private config: InitConfig) {
    env.allowLocalModels =
      config.transformersSettings?.allowLocalModels ?? false;
    env.useBrowserCache = config.transformersSettings?.useBrowserCache ?? false;

    this.config = { ...config, maxWidth: config.maxWidth || 12 };
  }

  async initialize(): Promise<void> {
    const { tokenizerPath, onnxSettings, maxWidth } = this.config;

    const tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);
    console.log("Tokenizer loaded.");
    const onnxWrapper = new ONNXWrapper(onnxSettings);

    const wordSplitter = new WhitespaceTokenSplitter();
    const processor = new SpanProcessor(
      { max_width: maxWidth },
      tokenizer,
      wordSplitter,
    );
    const decoder = new SpanDecoder({ max_width: maxWidth });

    this.model = new Model(
      { max_width: maxWidth },
      processor,
      decoder,
      onnxWrapper,
    );

    await this.model.initialize();
  }

  async inference({
    texts,
    entities,
    flatNer = false,
    threshold = 0.5,
  }: IInference): Promise<InferenceResultMultiple> {
    if (!this.model) {
      throw new Error("Model is not initialized. Call initialize() first.");
    }

    const result = await this.model.inference(
      texts,
      entities,
      flatNer,
      threshold,
    );
    return this.mapRawResultToResponse(result);
  }

  async inference_with_chunking({
    texts,
    entities,
    flatNer = false,
    threshold = 0.5,
  }: IInference): Promise<InferenceResultMultiple> {
    if (!this.model) {
      throw new Error("Model is not initialized. Call initialize() first.");
    }

    const result = await this.model.inference_with_chunking(
      texts,
      entities,
      flatNer,
      threshold,
    );
    return this.mapRawResultToResponse(result);
  }

  mapRawResultToResponse(
    rawResult: RawInferenceResult,
  ): InferenceResultMultiple {
    const response: InferenceResultMultiple = [];
    for (const individualResult of rawResult) {
      const entityResult: IEntityResult[] = individualResult.map(
        ([spanText, start, end, label, score]) => ({
          spanText,
          start,
          end,
          label,
          score,
        }),
      );
      response.push(entityResult);
    }

    return response;
  }
}
