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
    modelPath: "public/model.onnx", // Can be a string path or Uint8Array/ArrayBufferLike
    executionProvider: "webgpu", // Optional: "cpu", "wasm", "webgpu", or "webgl"
    wasmPaths: "path/to/wasm", // Optional: path to WASM binaries
    multiThread: true, // Optional: enable multi-threading (for wasm/cpu providers)
    maxThreads: 4, // Optional: specify number of threads (for wasm/cpu providers)
    fetchBinary: true, // Optional: prefetch binary from wasmPaths
  },
  transformersSettings: {
    // Optional
    allowLocalModels: true,
    useBrowserCache: true,
  },
  maxWidth: 12, // Optional
  modelType: "gliner", // Optional
});

await gliner.initialize();

const texts = ["Your input text here"];
const entities = ["city", "country", "person"];
const options = {
  flatNer: false, // Optional
  threshold: 0.1, // Optional
  multiLabel: false, // Optional
};

const results = await gliner.inference({
  texts,
  entities,
  ...options,
});
console.log(results);
```

### Response Format

The inference results will be returned in the following format:

```javascript
// For a single text input:
[
  {
    spanText: "New York", // The extracted entity text
    start: 10, // Start character position
    end: 18, // End character position
    label: "city", // Entity type
    score: 0.95, // Confidence score
  },
  // ... more entities
];

// For multiple text inputs, you'll get an array of arrays
```

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
