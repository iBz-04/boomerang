<script lang="ts">
	import { indexVideo, search, highlights, clipUrl, type ClipResult } from '$lib/api';

	let fileInput = $state<HTMLInputElement>();
	let videoEl = $state<HTMLVideoElement>();

	let videoUrl = $state('');
	let query = $state('');
	let status = $state<'idle' | 'indexing' | 'ready' | 'searching'>('idle');
	let result = $state<ClipResult | null>(null);
	let error = $state('');

	const busy = $derived(status === 'indexing' || status === 'searching');
	const hasVideo = $derived(videoUrl !== '');

	function fmt(t: number): string {
		const m = Math.floor(t / 60);
		const s = Math.floor(t % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	async function onFile(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		error = '';
		result = null;
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

	async function show(promise: Promise<{ results: ClipResult[] }>) {
		error = '';
		status = 'searching';
		try {
			const r = await promise;
			result = r.results[0] ?? null;
			if (result) {
				videoUrl = clipUrl(result.clip_url);
			} else {
				error = "No matches found for this query. Try being more descriptive.";
			}
			status = 'ready';
		} catch (e) {
			error = (e as Error).message;
			status = 'ready';
		}
	}

	function runSearch() {
		if (query.trim()) show(search(query));
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
				<video bind:this={videoEl} src={videoUrl} playsinline controls>
					<track kind="captions" />
				</video>
				{#if result}
					<button class="marker target" onclick={() => videoEl?.play()}>
						<span class="dot"></span>
						<span class="tag">TARGET · {fmt(result.start)}</span>
					</button>
					<span class="marker camera">
						<span class="tag">CAMERA · {Math.round(result.score * 100)}%</span>
					</span>
				{/if}
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

			<button class="btn primary" onclick={runSearch} disabled={busy || !hasVideo || !query.trim()}>
				<span class="ico">◎</span> SEARCH CLIP
			</button>
			<button class="btn ghost" onclick={() => show(highlights())} disabled={busy || !hasVideo}>
				<span class="ico">⤬</span> SURFACE HIGHLIGHTS
			</button>

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

	.marker {
		position: absolute;
		display: flex;
		align-items: center;
		gap: 8px;
		border: none;
		background: none;
		cursor: pointer;
		z-index: 2;
	}

	.target {
		top: 34%;
		right: 18%;
		flex-direction: column;
	}

	.target .dot {
		width: 40px;
		height: 40px;
		border-radius: 50%;
		background: #e23744;
		box-shadow: 0 0 0 6px rgba(226, 55, 68, 0.25);
	}

	.camera {
		bottom: 22%;
		left: 14%;
	}

	.tag {
		font-size: 11px;
		font-weight: 700;
		letter-spacing: 0.04em;
		color: #fff;
		background: rgba(0, 0, 0, 0.65);
		padding: 5px 10px;
		border-radius: 999px;
		white-space: nowrap;
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
