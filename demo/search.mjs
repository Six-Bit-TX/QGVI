/**
 * Small, dependency-free lexical search over the demo's stored captions.
 *
 * This is BM25 (k1 = 1.5, b = 0.75), not the complete evaluation index.
 * A scene filter defines the corpus before document frequencies and average
 * length are calculated. Query terms are unique; repeating a word does not
 * increase its influence. Only exact view IDs are deduplicated.
 *
 * Results are data, never HTML. Render every caption and query with textContent.
 */

const K1 = 1.5;
const B = 0.75;
const STOP_WORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by',
  'can', 'could', 'did', 'do', 'does', 'doing', 'for', 'from', 'had', 'has',
  'have', 'having', 'he', 'her', 'hers', 'him', 'his', 'how', 'i', 'if',
  'in', 'into', 'is', 'it', 'its', 'itself', 'me', 'my', 'of', 'on', 'or',
  'our', 'ours', 's', 'she', 'should', 'so', 'some', 'such', 't', 'than',
  'that', 'the', 'their', 'theirs', 'them', 'then', 'there', 'these', 'they',
  'this', 'those', 'to', 'too', 'us', 'was', 'we', 'were', 'what', 'when',
  'where', 'which', 'who', 'whom', 'will', 'with', 'would', 'you', 'your',
  'yours',
]);

/** Normalize Unicode words without discarding spatial, color, or number terms. */
export function tokenize(text) {
  if (typeof text !== 'string') return [];
  const words = text.normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  return words.filter((word) => !STOP_WORDS.has(word));
}

/**
 * Return at most `limit` matching documents, with positive finite BM25 scores.
 * Each output preserves its document fields and adds `score` and `matchedTerms`.
 * Equal scores sort by string ID in code-point order (not host locale).
 * Duplicate IDs retain their first occurrence in the selected corpus.
 */
export function rankViews(documents, query, { scene = 'all', limit = 6 } = {}) {
  const terms = [...new Set(tokenize(query))];
  if (!terms.length || !Array.isArray(documents)) return [];
  const resultLimit = Number.isFinite(limit) ? Math.max(0, Math.floor(limit)) : 6;
  if (!resultLimit) return [];

  const seen = new Set();
  const corpus = [];
  for (const document of documents) {
    if (!document || typeof document !== 'object' || document.id == null) continue;
    if (scene !== 'all' && document.scene !== scene) continue;
    const key = String(document.id);
    if (seen.has(key)) continue;
    seen.add(key);
    const tokens = tokenize(document.caption);
    const frequency = new Map();
    for (const token of tokens) frequency.set(token, (frequency.get(token) ?? 0) + 1);
    corpus.push({ document, key, length: tokens.length, frequency });
  }
  if (!corpus.length) return [];

  const averageLength = corpus.reduce((sum, row) => sum + row.length, 0) / corpus.length;
  if (!averageLength) return [];
  const documentFrequency = new Map(terms.map((term) => [term, 0]));
  for (const row of corpus) {
    for (const term of terms) {
      if (row.frequency.has(term)) {
        documentFrequency.set(term, documentFrequency.get(term) + 1);
      }
    }
  }

  const results = [];
  for (const row of corpus) {
    let score = 0;
    const matchedTerms = [];
    for (const term of terms) {
      const frequency = row.frequency.get(term) ?? 0;
      if (!frequency) continue;
      const df = documentFrequency.get(term);
      const inverseFrequency = Math.log(1 + (corpus.length - df + 0.5) / (df + 0.5));
      const normalizer = frequency + K1 * (1 - B + B * row.length / averageLength);
      score += inverseFrequency * (frequency * (K1 + 1)) / normalizer;
      matchedTerms.push(term);
    }
    if (score > 0 && Number.isFinite(score)) {
      results.push({ ...row.document, score, matchedTerms });
    }
  }
  results.sort((left, right) => {
    if (left.score !== right.score) return right.score - left.score;
    const leftId = String(left.id);
    const rightId = String(right.id);
    return leftId < rightId ? -1 : leftId > rightId ? 1 : 0;
  });
  return results.slice(0, resultLimit);
}
