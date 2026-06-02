// Typed client for the Boomerang Rust backend REST contract.
// No mock data: every call hits the real backend and surfaces errors.

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8080';

export interface MatchResult {
	file: string;
	start: number; // seconds
	end: number; // seconds
	score: number; // 0..1 cosine similarity
}

export interface SearchResponse {
	results: MatchResult[];
	rewritten_query?: string;
	search_queries?: string[];
}

export interface IndexResponse {
	file: string;
	chunks: number;
}

async function json<T>(res: Response): Promise<T> {
	if (!res.ok) {
		let message = `${res.status} ${res.statusText}`;
		const contentType = res.headers.get('content-type') ?? '';
		if (contentType.includes('application/json')) {
			const body = (await res.json()) as { error?: unknown };
			if (typeof body.error === 'string' && body.error.trim()) {
				message = body.error;
			}
		} else {
			const body = await res.text();
			if (body.trim()) {
				message = body;
			}
		}
		throw new Error(message);
	}
	return res.json() as Promise<T>;
}

// POST /index: multipart upload, chunks and embeds the video into the vector store.
export async function indexVideo(file: File): Promise<IndexResponse> {
	const body = new FormData();
	body.append('video', file);
	return json(await fetch(`${API_BASE}/index`, { method: 'POST', body }));
}

// POST /search: natural language query, returns ranked time ranges.
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
				threshold: opts.threshold ?? 0.3
			})
		})
	);
}

// POST /highlights: surface the most anomalous time ranges in the indexed footage.
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
