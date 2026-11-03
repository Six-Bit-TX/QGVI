import { rankViews } from './search.mjs';

const $ = (selector) => document.querySelector(selector);
const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const image = (src, alt, className) => {
  const node = make('img', className);
  node.src = src;
  node.alt = alt;
  node.loading = 'lazy';
  return node;
};
const download = (name, payload) => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json' }));
  const a = make('a');
  a.href = url;
  a.download = name.replace(/[^a-zA-Z0-9_.-]/g, '_');
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
};

// All routes use only this site's static assets. No model or rendering endpoint.
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab) {
  for (const candidate of tabs) {
    const active = candidate === tab;
    candidate.setAttribute('aria-selected', String(active));
    candidate.tabIndex = active ? 0 : -1;
    $(`#panel-${candidate.dataset.tab}`).hidden = !active;
  }
}
for (const tab of tabs) {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', (event) => {
    let i = tabs.indexOf(tab);
    if (event.key === 'ArrowRight') i = (i + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') i = (i + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') i = 0;
    else if (event.key === 'End') i = tabs.length - 1;
    else return;
    event.preventDefault();
    selectTab(tabs[i]);
    tabs[i].focus();
  });
}

$('#copy-citation').addEventListener('click', async () => {
  const text = $('#citation').textContent;
  try {
    await navigator.clipboard.writeText(text);
    $('#copy-status').textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents($('#citation'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    $('#copy-status').textContent = 'Citation selected. Press Ctrl+C or Command+C to copy.';
  }
});
const dialog = $('#view-dialog');
$('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });

async function init() {
  const response = await fetch('data/demo.json');
  if (!response.ok) throw new Error(`Demo data returned HTTP ${response.status}`);
  const data = await response.json();
  const views = data.text.views;
  const sceneNames = new Map(data.scenes.map((s) => [s.id, s.label]));
  $('#index-scope').textContent = `${views.length} selected views across ${data.scenes.length} scenes. Search their stored captions with BM25. This is a small demonstration index, not the full evaluation database.`;
  for (const scene of data.scenes) {
    const option = make('option', '', scene.label);
    option.value = scene.id;
    $('#scene').append(option);
  }
  function cameraPayload(view) {
    return { mode: 'stored_reference_camera', view_id: view.id, scene: view.scene,
      source_kind: view.source_kind, caption: view.caption, frame: view.frame,
      note: 'The stored camera is in the source scene coordinate frame. Intrinsics refer to source resolution, not the display thumbnail. This is not a newly estimated query pose.' };
  }
  function showView(view) {
    $('#dialog-image').src = view.image;
    $('#dialog-image').alt = view.alt || view.caption;
    $('#dialog-scene').textContent = `${sceneNames.get(view.scene) || view.scene} / ${view.source_kind === 'gaussian_grid_render' ? 'Gaussian grid render' : 'Original reference image'}`;
    $('#dialog-title').textContent = 'View + stored camera';
    $('#dialog-caption').textContent = view.caption;
    $('#dialog-download').onclick = () => download(`${view.id}-camera.json`, cameraPayload(view));
    dialog.showModal();
  }
  function search() {
    const query = $('#query').value.trim();
    const scene = $('#scene').value;
    const results = rankViews(views, query, { scene, limit: 6 });
    const target = $('#search-results');
    target.replaceChildren();
    const count = views.filter((v) => scene === 'all' || v.scene === scene).length;
    $('#search-status').textContent = `${results.length} ${results.length === 1 ? 'view' : 'views'} shown · ${count} captions searched`;
    if (!results.length) {
      target.append(make('div', 'empty-state', query ? 'No matching caption in this small index. Try different words or another scene.' : 'Describe a view, or choose one of the example queries above.'));
      return;
    }
    results.forEach((view, i) => {
      const card = make('button', 'view-card');
      card.type = 'button';
      card.setAttribute('aria-label', `Open result ${i + 1}: ${sceneNames.get(view.scene)}. ${view.caption}`);
      card.append(image(view.image, view.alt || view.caption));
      const body = make('div', 'view-card-body');
      const meta = make('div', 'view-card-meta');
      meta.append(make('span', '', `${String(i + 1).padStart(2, '0')} / ${sceneNames.get(view.scene)}`), make('span', '', `BM25 ${view.score.toFixed(2)}`));
      body.append(meta, make('p', '', view.caption), make('span', 'view-type', `${view.source_kind === 'gaussian_grid_render' ? 'Gaussian grid render' : 'Original reference image'} · Camera details ↗`));
      card.append(body);
      card.addEventListener('click', () => showView(view));
      target.append(card);
    });
  }
  $('#search-form').addEventListener('submit', (event) => { event.preventDefault(); search(); });
  $('#scene').addEventListener('change', search);
  for (const preset of data.text.presets) {
    const button = make('button', '', preset.label);
    button.type = 'button';
    button.addEventListener('click', () => {
      $('#query').value = preset.query;
      $('#scene').value = preset.scene || 'all';
      search();
    });
    $('#query-presets').append(button);
  }
  $('#query').value = data.text.presets[0]?.query || '';
  $('#scene').value = data.text.presets[0]?.scene || 'all';
  search();

  function renderRgb(example) {
    for (const button of $('#rgb-choices').children) button.setAttribute('aria-pressed', String(button.dataset.id === example.id));
    const layout = make('div', 'rgb-layout');
    const query = make('figure');
    query.append(image(example.query_image, example.nature), make('figcaption', '', `Query / ${sceneNames.get(example.source_scene)}`));
    const reference = make('figure');
    if (example.reference_image) {
      reference.append(image(example.reference_image, example.reference_label), make('figcaption', '', example.reference_label));
    } else {
      const empty = make('div', 'rgb-no-result');
      empty.append(make('span', '', '×'), make('h3', '', 'No accepted pose'), make('p', '', 'A PnP hypothesis exists, but the evidence fails the gate for the SF map.'));
      reference.append(empty, make('figcaption', '', 'Recorded rejection / SF map'));
    }
    const detail = make('div', 'rgb-details');
    detail.append(make('span', `accept-label${example.accepted ? '' : ' reject'}`, example.accepted ? '✓ Geometric pose accepted' : '× Geometric pose rejected'));
    detail.append(make('div', 'keypoint-count', example.distinct_keypoints.toLocaleString()), make('p', 'keypoint-caption', 'distinct inlier query keypoints'));
    const meter = make('div', `keypoint-meter${example.accepted ? '' : ' reject'}`);
    const bar = make('span');
    bar.style.width = `${Math.min(100, example.distinct_keypoints / data.rgb.threshold * 100)}%`;
    meter.append(bar);
    detail.append(meter, make('p', 'small-note', `${example.distinct_keypoints.toLocaleString()} ${example.accepted ? '≥' : '<'} ${data.rgb.threshold} · fixed acceptance threshold. ${example.notes}`), make('p', 'small-note', example.nature));
    const button = make('button', 'text-button', example.accepted ? 'Download recorded pose JSON ↓' : 'Download recorded decision JSON ↓');
    button.addEventListener('click', () => download(`qgvi-rgb-${example.id}.json`, {
      mode: 'cached_geometric_localization', id: example.id, accepted: example.accepted,
      map_scene: example.map_scene, source_scene: example.source_scene,
      distinct_keypoints: example.distinct_keypoints, threshold: data.rgb.threshold,
      c2w: example.c2w, coordinate_units: example.coordinate_units, note: example.notes,
    }));
    detail.append(button);
    layout.append(query, reference, detail);
    $('#rgb-content').replaceChildren(layout);
  }
  for (const example of data.rgb.examples) {
    const button = make('button', '', example.label);
    button.dataset.id = example.id;
    button.addEventListener('click', () => renderRgb(example));
    $('#rgb-choices').append(button);
  }
  renderRgb(data.rgb.examples[0]);

  const agent = data.agent;
  let current = 0;
  $('#agent-source').src = agent.source_image;
  $('#agent-instruction').textContent = agent.instruction;
  for (const predicate of agent.predicates) $('#agent-predicates').append(make('li', '', `${predicate.id}: ${predicate.text}`));
  $('#agent-scope').textContent = `${agent.outcome} ${agent.review_scope} This episode used caption search and camera exploration; it did not call RGB localization.`;
  function step(index) {
    current = Math.max(0, Math.min(agent.steps.length - 1, index));
    const observation = agent.steps[current];
    $('#agent-frame').src = observation.image || agent.source_image;
    $('#agent-frame').alt = `Recorded observation ${observation.index}: ${observation.summary}`;
    $('#agent-frame-label').textContent = `${String(observation.index).padStart(2, '0')} / ${observation.action} · saved observation`;
    $('#agent-step-count').textContent = `DECISION ${current + 1} OF ${agent.steps.length}`;
    $('#agent-step-title').textContent = observation.title;
    $('#agent-step-summary').textContent = observation.summary;
    $('#agent-checks').replaceChildren();
    if (observation.predicate_results?.length) {
      const checks = make('ul', 'check-list');
      observation.predicate_results.forEach((check) => {
        const satisfied = check.status === 'satisfied';
        const li = make('li');
        const text = make('span', '', `${check.predicate_id}: ${check.status}`);
        text.title = check.reason;
        li.append(make('span', `check-symbol${satisfied ? '' : ' no'}`, satisfied ? '✓' : '×'), text);
        checks.append(li);
      });
      $('#agent-checks').append(checks, make('p', 'small-note', 'Recorded controller assessments.'));
    } else if (observation.unresolved_requirements?.length) {
      $('#agent-checks').append(make('p', 'small-note', `Still unresolved: ${observation.unresolved_requirements.join(', ')}.`));
    }
    $('#agent-prev').disabled = current === 0;
    $('#agent-next').disabled = current === agent.steps.length - 1;
    [...$('#agent-timeline').children].forEach((button, i) => {
      if (i === current) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
  }
  agent.steps.forEach((observation, i) => {
    const button = make('button', '', String(i + 1));
    button.setAttribute('aria-label', `Decision ${i + 1}: ${observation.title}`);
    button.addEventListener('click', () => step(i));
    $('#agent-timeline').append(button);
  });
  $('#agent-prev').addEventListener('click', () => step(current - 1));
  $('#agent-next').addEventListener('click', () => step(current + 1));
  $('#agent-pose').addEventListener('click', () => download('qgvi-agent-final-camera.json', {
    mode: 'cached_agent_return', query_id: agent.id, scene: agent.scene,
    instruction: agent.instruction, frame: agent.final_frame,
    assessment: agent.outcome, assessment_scope: agent.review_scope,
  }));
  step(0);
  $('#load-status').textContent = '';
}
init().catch((error) => {
  $('#load-status').textContent = 'The demo assets could not be loaded. Reload this page to try again. The paper and results remain available below.';
  console.error('Demo initialization failed:', error);
});
