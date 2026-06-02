// Typed client for the Boomerang Rust backend REST contract.
// No mock data: every call hits the real backend and surfaces errors.

import {
	API_BASE,
	HIGHLIGHTS_COUNT,
	HIGHLIGHTS_METHOD,
	SEARCH_DEDUPE_THRESHOLD,
	SEARCH_RESULTS,
	SEARCH_THRESHOLD
} from '$lib/config';

export interface MatchResult {
	file: string;
	source_file: string;
	start: number;
	end: number;
	score: number;
}

export interface SearchResponse {
	results: MatchResult[];
	rewritten_query?: string;
	search_queries?: string[];
}

export interface IndexResponse {
	file: string;
	source_file: string;
	chunks: number;
}

async function json<T>(res: Response): Promise<T> {
	if (!res.ok) {
		let message = `${res.status} ${res.statusText}`;
		const contentType = res.headers.get('content-type');
		if (contentType?.includes('application/json')) {
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

export async function indexVideo(file: File): Promise<IndexResponse> {
	const body = new FormData();
	body.append('video', file);
	return json(await fetch(`${API_BASE}/index`, { method: 'POST', body }));
}

export async function search(query: string, sourceFile?: string): Promise<SearchResponse> {
	return json(
		await fetch(`${API_BASE}/search`, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({
				query,
				results: SEARCH_RESULTS,
				threshold: SEARCH_THRESHOLD,
				dedupe_threshold: SEARCH_DEDUPE_THRESHOLD,
				source_file: sourceFile
			})
		})
	);
}

export async function highlights(sourceFile?: string): Promise<SearchResponse> {
	return json(
		await fetch(`${API_BASE}/highlights`, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({
				count: HIGHLIGHTS_COUNT,
				method: HIGHLIGHTS_METHOD,
				source_file: sourceFile
			})
		})
	);
}
