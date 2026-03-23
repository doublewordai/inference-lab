use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::path::Path;

/// A tokenizer function that takes messages and returns tokenized output.
/// This allows different implementations (tiktoken, transformers.js, etc.)
/// to be passed in from the CLI or WASM interface.
/// The tokenizer should apply the appropriate chat template.
pub type TokenizerFn = Box<dyn Fn(&[Message]) -> Result<Vec<u32>, String> + Send + Sync>;

/// A batch tokenizer function that takes multiple message arrays and returns multiple token vectors.
/// This is much faster than tokenizing one at a time.
pub type BatchTokenizerFn = Box<dyn Fn(&[&[Message]]) -> Result<Vec<Vec<u32>>, String> + Send>;

/// OpenAI Batch API format - JSONL entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    pub custom_id: String,
    pub method: String,
    pub url: String,
    pub body: RequestBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestBody {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A processed dataset entry ready for simulation
#[derive(Debug, Clone)]
pub struct DatasetEntry {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub max_output_tokens: Option<u32>,
}

impl DatasetEntry {
    pub fn num_prompt_tokens(&self) -> u32 {
        self.prompt_tokens.len() as u32
    }
}

/// Unparsed entry from dataset (before tokenization)
#[derive(Debug, Clone)]
pub struct UnparsedEntry {
    pub request_id: String,
    pub messages: Vec<Message>,
    pub max_output_tokens: Option<u32>,
}

/// Iterator over dataset entries, parsing JSON but NOT tokenizing
/// Tokenization happens in batches in the background thread for performance
pub struct DatasetIterator<R: BufRead> {
    lines: Lines<R>,
    line_num: usize,
    sent_end_signal: bool,
}

impl<R: BufRead> DatasetIterator<R> {
    pub fn new(reader: R) -> Self {
        Self {
            lines: reader.lines(),
            line_num: 0,
            sent_end_signal: false,
        }
    }
}

impl<R: BufRead> Iterator for DatasetIterator<R> {
    type Item = Result<Option<UnparsedEntry>, Box<dyn std::error::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.line_num += 1;
            let line = match self.lines.next() {
                Some(Ok(line)) => line,
                Some(Err(e)) => return Some(Err(Box::new(e))),
                None => {
                    // End of dataset - signal completion with Ok(None) once, then end iterator
                    if !self.sent_end_signal {
                        self.sent_end_signal = true;
                        return Some(Ok(None));
                    } else {
                        return None;
                    }
                }
            };

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            // Parse the batch request (but don't tokenize yet - that happens in batches)
            let batch_request: BatchRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    return Some(Err(format!(
                        "Failed to parse line {}: {}",
                        self.line_num, e
                    )
                    .into()))
                }
            };

            return Some(Ok(Some(UnparsedEntry {
                request_id: batch_request.custom_id,
                messages: batch_request.body.messages,
                max_output_tokens: batch_request.body.max_tokens,
            })));
        }
    }
}

/// Dataset loader that provides lazy iteration over entries
pub struct DatasetLoader;

impl DatasetLoader {
    /// Count the number of non-empty lines in a JSONL file (fast approximation of entry count)
    pub fn count_entries<P: AsRef<Path>>(path: P) -> Result<usize, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let count = reader
            .lines()
            .filter_map(|line| line.ok())
            .filter(|line| !line.trim().is_empty())
            .count();
        Ok(count)
    }

    /// Create an iterator from a JSONL file in OpenAI batch API format
    /// Returns unparsed entries (without tokenization)
    pub fn from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<DatasetIterator<BufReader<File>>, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(DatasetIterator::new(reader))
    }

    /// Create an iterator from a string (useful for testing or WASM)
    /// Returns unparsed entries (without tokenization)
    pub fn from_string(data: String) -> DatasetIterator<BufReader<std::io::Cursor<String>>> {
        let cursor = std::io::Cursor::new(data);
        let reader = BufReader::new(cursor);
        DatasetIterator::new(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_batch_request() {
        let json = r#"{
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 100
            }
        }"#;

        let batch_request: BatchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(batch_request.custom_id, "request-1");
        assert_eq!(batch_request.body.messages.len(), 2);
        assert_eq!(batch_request.body.max_tokens, Some(100));
    }

    #[test]
    fn test_dataset_iterator() {
        let test_data = r#"{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "World"}], "max_tokens": 20}}"#;

        let mut iter = DatasetLoader::from_string(test_data.to_string());

        let entry1 = iter.next().unwrap().unwrap().unwrap();
        assert_eq!(entry1.request_id, "req-1");
        assert_eq!(entry1.messages.len(), 1);
        assert_eq!(entry1.messages[0].role, "user");
        assert_eq!(entry1.messages[0].content, "Hello");
        assert_eq!(entry1.max_output_tokens, Some(10));

        let entry2 = iter.next().unwrap().unwrap().unwrap();
        assert_eq!(entry2.request_id, "req-2");
        assert_eq!(entry2.max_output_tokens, Some(20));

        // Should get Ok(None) signaling end of dataset
        let end_signal = iter.next().unwrap().unwrap();
        assert!(end_signal.is_none());

        // After that, iterator itself should be exhausted
        assert!(iter.next().is_none());
    }
}
