# 👑 GLiNER.js: Generalist and Lightweight Named Entity Recognition for JavaScript

GLiNER.js is a TypeScript-based inference engine for running GLiNER (Generalist and Lightweight Named Entity Recognition) models. GLiNER can identify any entity type using a bidirectional transformer encoder, offering a practical alternative to traditional NER models and large language models.

<p align="center">
    <a href="https://arxiv.org/abs/2311.08526">📄 Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/Y2yVxpSQnG">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/spaces/urchade/gliner_mediumv2.1">🤗 Demo</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/models?library=gliner&sort=trending">🤗 Available models</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://github.com/urchade/GLiNER">🧬 Official Repo</a>
</p>

## 🌟 Key Features

- Flexible entity recognition without predefined categories
- Lightweight and fast inference
- Easy integration with web applications
- TypeScript support for better developer experience

## 🚀 Getting Started

### Installation

```bash
npm install gliner
```

### Basic Usage

```javascript
const gliner = new Gliner({
  tokenizerPath: "onnx-community/gliner_small-v2",
  onnxSettings: {
    modelPath: "public/model.onnx",
    executionContext: "web",
    executionProvider: "webgpu",
  },
  maxWidth: 12,
});

await gliner.initialize();

const input_text = "Your input text here";
const texts = [input_text];
const entities = ["city", "country", "person"];
const threshold = 0.1;

const decoded = await gliner.inference({ texts, entities, threshold });
console.log(decoded);
```

### Advanced Usage

#### ONNX settings API

- modelPath: can be either a URL to a local model as in the basic example, or it can also be the Model itself as an array of binary data.
- executionContext: options here are `node` or `web`
- executionProvider: these are the same providers that ONNX web supports, currently we allow `webgpu` (recommended), `cpu`, `wasm`, `webgl` but more can be added
- wasmPaths: Path to the wasm binaries, this can be either a URL to the binaries like a CDN url, or a local path to a folder with the binaries.
- multiThread: wether to multithread at all, only relevent for wasm and cpu exeuction providers.
- multiThread: When choosing the wasm or cpu provider and the web context, multiThread will allow you to specify the number of cores you want to use.
- fetchBinary: will prefetch the binary from the default or provided wasm paths

## 🛠 Setup & Model Preparation

To use GLiNER models in a web environment, you need an ONNX format model. You can:

1. Search for pre-converted models on [HuggingFace](https://huggingface.co/onnx-community?search_models=gliner)
2. Convert a model yourself using the [official Python script](https://github.com/urchade/GLiNER/blob/main/convert_to_onnx.py)

### Converting to ONNX Format

Use the `convert_to_onnx.py` script with the following arguments:

- `model_path`: Location of the GLiNER model
- `save_path`: Where to save the ONNX file
- `quantize`: Set to True for IntU8 quantization (optional)

Example:

```bash
python convert_to_onnx.py --model_path /path/to/your/model --save_path /path/to/save/onnx --quantize True
```

## 🌟 Use Cases

GLiNER.js offers versatile entity recognition capabilities across various domains:

1. **Enhanced Search Query Understanding**
2. **Real-time PII Detection**
3. **Intelligent Document Parsing**
4. **Content Summarization and Insight Extraction**
5. **Automated Content Tagging and Categorization**
   ...

## 🔧 Areas for Improvement

- [ ] Further optimize inference speed
- [ ] Add support for token-based GLiNER architecture
- [ ] Implement bi-encoder GLiNER architecture for better scalability
- [ ] Enable model training capabilities
- [ ] Provide more usage examples

## Creating a PR

- for any changes, remember to run `pnpm changeset`, otherwise there will not be a version bump and the PR Github Action will fail.

## 🙏 Acknowledgements

- [GLiNER original authors](https://github.com/urchade/GLiNER)
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)
- [Transformers.js](https://github.com/xenova/transformers.js)

## 📞 Support

For questions and support, please join our [Discord community](https://discord.gg/ApZvyNZU) or open an issue on GitHub.
