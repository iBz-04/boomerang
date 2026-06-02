function requireEnv(name: string): string {
	const value = import.meta.env[name];
	if (typeof value !== 'string' || !value.trim()) {
		throw new Error(`${name} is not configured`);
	}
	return value.trim();
}

function requirePositiveInt(name: string): number {
	const raw = requireEnv(name);
	const value = Number(raw);
	if (!Number.isInteger(value) || value < 1) {
		throw new Error(`${name} must be a positive integer, got ${raw}`);
	}
	return value;
}

function requireUnitInterval(name: string): number {
	const raw = requireEnv(name);
	const value = Number(raw);
	if (!Number.isFinite(value) || value < 0 || value > 1) {
		throw new Error(`${name} must be a number between 0 and 1, got ${raw}`);
	}
	return value;
}

export const API_BASE = requireEnv('VITE_API_BASE');
export const SEARCH_RESULTS = requirePositiveInt('VITE_SEARCH_RESULTS');
export const SEARCH_THRESHOLD = requireUnitInterval('VITE_SEARCH_THRESHOLD');
export const SEARCH_DEDUPE_THRESHOLD = requireUnitInterval('VITE_SEARCH_DEDUPE_THRESHOLD');
export const HIGHLIGHTS_COUNT = requirePositiveInt('VITE_HIGHLIGHTS_COUNT');
export const HIGHLIGHTS_METHOD = requireEnv('VITE_HIGHLIGHTS_METHOD');
