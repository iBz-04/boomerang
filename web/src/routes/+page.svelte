<script lang="ts">
	import { indexVideo, search, highlights, type MatchResult } from '$lib/api';

	let fileInput = $state<HTMLInputElement>();
	let videoEl = $state<HTMLVideoElement>();

	let videoUrl = $state('');
	let query = $state('');
	let status = $state<'idle' | 'indexing' | 'ready' | 'searching'>('idle');
	let results = $state<MatchResult[]>([]);
	let rewrittenQuery = $state('');
	let searchQueries = $state<string[]>([]);
	let error = $state('');

	const busy = $derived(status === 'indexing' || status === 'searching');
	const hasVideo = $derived(videoUrl !== '');

	function fmt(t: number): string {
		const m = Math.floor(t / 60);
		const s = Math.floor(t % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	async function seekToResult(match: MatchResult) {
		error = '';
		if (!videoEl) {
			throw new Error('video element is not ready');
		}
		if (!Number.isFinite(match.start) || match.start < 0) {
			throw new Error(`invalid search result start time: ${match.start}`);
		}
		if (Number.isFinite(videoEl.duration) && match.start > videoEl.duration) {
			throw new Error(`search result start time exceeds video duration: ${match.start}`);
		}

		videoEl.currentTime = match.start;
		await videoEl.play();
	}

	async function onResultClick(match: MatchResult) {
		try {
			await seekToResult(match);
		} catch (e) {
			error = (e as Error).message;
		}
	}

	async function onFile(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		error = '';
		results = [];
		rewrittenQuery = '';
		searchQueries = [];
		videoUrl = URL.createObjectURL(file);
		status = 'indexing';
		try {
			await indexVideo(file);
			status = 'ready';
		} catch (e) {
			error = (e as Error).message;
			status = 'idle';
		}
	}

	async function show(
		promise: Promise<{
			results: MatchResult[];
			rewritten_query?: string;
			search_queries?: string[];
		}>,
		opts: { autoPlayBest?: boolean } = {}
	) {
		error = '';
		status = 'searching';
		try {
			const r = await promise;
			results = r.results;
			searchQueries = r.search_queries ?? [];
			rewrittenQuery = r.rewritten_query?.trim() ?? '';
			if (results.length === 0) {
				error = 'No matches found. Try naming objects, actions, or colors you expect in the clip.';
			} else if (opts.autoPlayBest) {
				await onResultClick(results[0]);
			}
			status = 'ready';
		} catch (e) {
			error = (e as Error).message;
			status = 'ready';
		}
	}

	function runSearch() {
		if (query.trim()) show(search(query), { autoPlayBest: true });
	}
</script>

<main>
	<section class="card">
		<div class="media">
			<input
				bind:this={fileInput}
				type="file"
				accept="video/*"
				hidden
				onchange={onFile}
			/>

			<span class="badge logo">B</span>

			{#if hasVideo}
				{#key videoUrl}
					<video bind:this={videoEl} src={videoUrl} playsinline controls>
						<track kind="captions" />
					</video>
				{/key}
			{:else}
				<button class="prompt" onclick={() => fileInput?.click()}>
					<span class="cam-icon">⬡</span>
					<p>Upload footage</p>
					<small>tap to choose a video</small>
				</button>
			{/if}

			{#if busy}
				<div class="overlay"><span class="spinner"></span></div>
			{/if}
		</div>

		<div class="sheet">
			<h1>Boomerang</h1>

			<input
				class="query"
				type="text"
				placeholder="Describe the moment you want…"
				bind:value={query}
				onkeydown={(e) => e.key === 'Enter' && runSearch()}
				disabled={!hasVideo}
			/>
			<p class="label">{status === 'indexing' ? 'INDEXING FOOTAGE' : 'SEMANTIC QUERY'}</p>
			{#if searchQueries.length > 0}
				<ul class="queries" aria-label="Expanded search queries">
					{#each searchQueries as q, index}
						<li>{index + 1}. {q}</li>
					{/each}
				</ul>
			{:else if rewrittenQuery}
				<p class="rewrite">Searching as: {rewrittenQuery}</p>
			{/if}

			<button class="btn primary" onclick={runSearch} disabled={busy || !hasVideo || !query.trim()}>
				<span class="ico">◎</span> SEARCH TIME
			</button>
			<button class="btn ghost" onclick={() => show(highlights())} disabled={busy || !hasVideo}>
				<span class="ico">⤬</span> SURFACE HIGHLIGHTS
			</button>

			{#if results.length > 0}
				<div class="results" aria-label="Search results">
					{#each results as match, index}
						<button class="result" onclick={() => onResultClick(match)}>
							<span>{index + 1}. {fmt(match.start)}–{fmt(match.end)}</span>
							<span class="score">{Math.round(match.score * 100)}%</span>
						</button>
					{/each}
				</div>
			{/if}

			{#if error}<p class="error">{error}</p>{/if}
		</div>
	</section>
</main>

<style>
	:global(body) {
		margin: 0;
		background: #fff;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
		color: #111;
	}

	main {
		min-height: 100vh;
		display: grid;
		place-items: center;
		padding: 24px;
		box-sizing: border-box;
	}

	.card {
		width: 100%;
		max-width: 380px;
	}

	.media {
		position: relative;
		aspect-ratio: 3 / 4;
		border-radius: 12px;
		overflow: hidden;
		background: #f2f2f4;
	}

	.media video {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}

	.badge {
		position: absolute;
		display: grid;
		place-items: center;
		font-weight: 700;
		font-size: 13px;
	}

	.logo {
		top: 16px;
		left: 16px;
		width: 34px;
		height: 34px;
		border-radius: 50%;
		background: #fff;
		border: 1px solid #e3e3e6;
		z-index: 3;
	}

	.prompt {
		position: absolute;
		inset: 0;
		display: grid;
		place-content: center;
		justify-items: center;
		gap: 4px;
		color: #6b6b70;
		border: none;
		background: none;
		cursor: pointer;
		font: inherit;
	}

	.prompt .cam-icon {
		font-size: 40px;
	}

	.prompt p {
		margin: 4px 0 0;
		font-weight: 600;
		color: #333;
	}

	.prompt small {
		font-size: 12px;
	}

	.overlay {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		background: rgba(255, 255, 255, 0.5);
		z-index: 4;
	}

	.spinner {
		width: 32px;
		height: 32px;
		border: 3px solid rgba(0, 0, 0, 0.15);
		border-top-color: #111;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.sheet {
		margin-top: 16px;
		position: relative;
		background: #fff;
		padding: 4px 0 0;
		display: grid;
		gap: 12px;
	}

	h1 {
		margin: 0;
		text-align: center;
		font-size: 22px;
		font-weight: 700;
	}

	.query {
		width: 100%;
		box-sizing: border-box;
		border: 1px solid #e3e3e6;
		background: #fff;
		border-radius: 8px;
		padding: 14px 16px;
		font-size: 15px;
		outline: none;
	}

	.query:focus {
		border-color: #111;
	}

	.query:disabled {
		opacity: 0.5;
	}

	.label {
		margin: 0 0 2px 4px;
		font-size: 11px;
		font-weight: 700;
		letter-spacing: 0.12em;
		color: #9a9aa0;
	}

	.rewrite {
		margin: -6px 0 2px 4px;
		font-size: 12px;
		color: #64646b;
	}

	.queries {
		margin: -6px 0 2px 0;
		padding: 0 0 0 18px;
		font-size: 12px;
		color: #64646b;
		display: grid;
		gap: 4px;
	}

	.btn {
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		border: 1px solid transparent;
		border-radius: 8px;
		padding: 9px 14px;
		font-size: 13px;
		font-weight: 600;
		letter-spacing: 0.06em;
		cursor: pointer;
	}

	.btn:disabled {
		opacity: 0.45;
		cursor: default;
	}

	.primary {
		background: #111;
		color: #fff;
	}

	.ghost {
		background: #fff;
		color: #111;
		border-color: #e3e3e6;
	}

	.ico {
		font-size: 14px;
	}

	.results {
		display: grid;
		gap: 8px;
		margin-top: 2px;
	}

	.result {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		width: 100%;
		border: 1px solid #e3e3e6;
		border-radius: 8px;
		background: #fff;
		color: #111;
		padding: 11px 12px;
		font: inherit;
		font-size: 13px;
		cursor: pointer;
		text-align: left;
	}

	.result:hover {
		border-color: #111;
	}

	.score {
		font-size: 11px;
		font-weight: 700;
		color: #6b6b70;
		flex-shrink: 0;
	}

	.error {
		margin: 0;
		text-align: center;
		font-size: 12px;
		color: #e23744;
	}

	/* Desktop: two-column layout filling the width, side control panel. */
	@media (min-width: 820px) {
		main {
			padding: 40px;
		}

		.card {
			max-width: none;
			height: calc(100vh - 80px);
			display: grid;
			grid-template-columns: 1.4fr 1fr;
			gap: 40px;
			align-items: stretch;
		}

		.media {
			aspect-ratio: auto;
			height: 100%;
		}

		.sheet {
			margin-top: 0;
			align-content: center;
			gap: 16px;
			padding: 0 0 0 12px;
		}

		h1 {
			text-align: left;
			font-size: 30px;
		}

		.label {
			margin-left: 2px;
		}
	}
</style>
