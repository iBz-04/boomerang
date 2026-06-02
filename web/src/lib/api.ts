// Typed client for the Boomerang Rust backend REST contract.
// No mock data: every call hits the real backend and surfaces errors.

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080';

export interface ClipResult {
	file: string;
	start: number; // seconds
	end: number; // seconds
	score: number; // 0..1 cosine similarity
	clip_url: string; // path or absolute url to the trimmed clip
}

export interface SearchResponse {
	results: ClipResult[];
}

export interface IndexResponse {
	file: string;
	chunks: number;
}

export function clipUrl(path: string): string {
	return path.startsWith('http') ? path : `${API_BASE}${path}`;
}

async function json<T>(res: Response): Promise<T> {
	if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
	return res.json() as Promise<T>;
}

// POST /index — multipart upload, chunks + embeds the video into the vector store.
export async function indexVideo(file: File): Promise<IndexResponse> {
	const body = new FormData();
	body.append('video', file);
	return json(await fetch(`${API_BASE}/index`, { method: 'POST', body }));
}

// POST /search — natural language query, returns ranked clips.
export async function search(
	query: string,
	opts: { results?: number; threshold?: number } = {}
): Promise<SearchResponse> {
	return json(
		await fetch(`${API_BASE}/search`, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({
				query,
				results: opts.results ?? 5,
				threshold: opts.threshold ?? 0.41
			})
		})
	);
}

// POST /highlights — surface the most anomalous clips in the indexed footage.
export async function highlights(
	opts: { count?: number; method?: string } = {}
): Promise<SearchResponse> {
	return json(
		await fetch(`${API_BASE}/highlights`, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ count: opts.count ?? 5, method: opts.method ?? 'knn' })
		})
	);
}
