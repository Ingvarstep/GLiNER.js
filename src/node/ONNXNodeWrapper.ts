import * as ort from "onnxruntime-node";
import { IONNXNodeSettings, IONNXNodeWrapper } from "../interfaces";

export class ONNXNodeWrapper implements IONNXNodeWrapper {
  public ort: typeof ort = ort;
  private session: ort.InferenceSession | null = null;

  constructor(public settings: IONNXNodeSettings) {}

  public async init() {
    // @ts-ignore
    this.session = await this.ort.InferenceSession.create(this.settings.modelPath);
  }

  public async run(
    feeds: ort.InferenceSession.FeedsType,
    options: ort.InferenceSession.RunOptions = {},
  ): Promise<ort.InferenceSession.ReturnType> {
    if (!this.session) {
      throw new Error("ONNXWrapper: Session not initialized. Please call init() first.");
    }

    return await this.session.run(feeds, options);
  }
}
