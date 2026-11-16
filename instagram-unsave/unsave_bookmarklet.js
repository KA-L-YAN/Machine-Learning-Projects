// Unsave all saved Instagram posts from the browser (works on a phone).
// Run it while logged in on https://www.instagram.com — it uses your existing
// browser session, so no password is ever entered or sent anywhere.
// Saved audio/music lives in a separate list and is not touched.
(async () => {
  if (!/(^|\.)instagram\.com$/.test(location.hostname)) {
    alert('Open instagram.com in this browser, log in, then run this again.');
    return;
  }
  if (document.getElementById('us_box')) return;
  const csrf = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
  const H = { 'X-IG-App-ID': '936619743392459', 'X-CSRFToken': csrf, 'X-Requested-With': 'XMLHttpRequest' };
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  const box = document.createElement('div');
  box.id = 'us_box';
  box.style.cssText = 'position:fixed;left:10px;right:10px;bottom:calc(10px + env(safe-area-inset-bottom,0px));z-index:2147483647;background:#fff;color:#111;border-radius:14px;padding:16px;font:15px/1.4 -apple-system,system-ui,sans-serif;box-shadow:0 6px 30px rgba(0,0,0,.35)';
  const btn = 'flex:1;padding:12px;border:0;border-radius:10px;font:600 15px system-ui;';
  box.innerHTML =
    '<div style="font-weight:700;font-size:16px">Unsave saved posts</div>' +
    '<div id="us_s" style="margin:8px 0 10px">Counting your saved posts…</div>' +
    '<div style="height:8px;background:#eee;border-radius:4px;overflow:hidden"><div id="us_b" style="height:100%;width:0;background:#0095f6;transition:width .3s"></div></div>' +
    '<div style="display:flex;gap:8px;margin-top:12px">' +
    '<button id="us_go" disabled style="' + btn + 'background:#0095f6;color:#fff;opacity:.5">Unsave all</button>' +
    '<button id="us_x" style="' + btn + 'background:#efefef;color:#111">Close</button></div>';
  document.body.appendChild(box);
  const $ = id => document.getElementById(id);
  const status = t => ($('us_s').textContent = t);
  let stop = false;
  $('us_x').onclick = () => { stop = true; box.remove(); };

  // 1. Collect every saved post id (paginated feed).
  const ids = [];
  try {
    let maxId = '';
    for (let page = 0; page < 500; page++) {
      const r = await fetch('/api/v1/feed/saved/posts/' + (maxId ? '?max_id=' + encodeURIComponent(maxId) : ''),
        { headers: H, credentials: 'include' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const j = await r.json();
      for (const it of j.items || []) if (it.media && it.media.pk) ids.push(String(it.media.pk));
      status('Counting your saved posts… ' + ids.length);
      if (!j.more_available || !j.next_max_id) break;
      maxId = j.next_max_id;
      await sleep(700);
    }
  } catch (e) {
    status('Could not read your saved posts (' + e.message + '). Make sure you are logged in on instagram.com and try again.');
    return;
  }
  if (!ids.length) { status('You have no saved posts.'); return; }

  status('Found ' + ids.length + ' saved posts. Tap "Unsave all" to remove them. Keep this tab open.');
  const go = $('us_go');
  go.disabled = false; go.style.opacity = 1;

  async function unsave(pk) {
    for (const url of ['/api/v1/web/save/' + pk + '/unsave/', '/web/save/' + pk + '/unsave/']) {
      const r = await fetch(url, {
        method: 'POST', credentials: 'include',
        headers: Object.assign({ 'Content-Type': 'application/x-www-form-urlencoded' }, H),
      });
      if (r.status === 404) continue;
      if (r.status === 429) throw new Error('rate');
      const j = await r.json().catch(() => ({}));
      if (j.feedback_required || j.spam) throw new Error('rate');
      return r.ok && j.status === 'ok';
    }
    return false;
  }

  // 2. Unsave them one by one, slowly, to avoid Instagram's rate limits.
  go.onclick = async () => {
    go.disabled = true; go.style.opacity = .5;
    $('us_x').textContent = 'Stop';
    $('us_x').onclick = () => { stop = true; };
    let done = 0, failed = 0;
    for (let i = 0; i < ids.length && !stop; i++) {
      try { (await unsave(ids[i])) ? done++ : failed++; }
      catch (e) {
        status('Instagram is limiting requests. Stopped after ' + done + ' unsaved. Wait an hour, reload, and run it again.');
        break;
      }
      $('us_b').style.width = ((i + 1) / ids.length * 100) + '%';
      status('Unsaved ' + done + ' of ' + ids.length + (failed ? ' (' + failed + ' failed)' : '') + '… keep this tab open.');
      await sleep(1500 + Math.random() * 1500);
    }
    if (!/limiting/.test($('us_s').textContent))
      status((stop ? 'Stopped. ' : 'Done. ') + 'Unsaved ' + done + ' post' + (done === 1 ? '' : 's') + (failed ? ', ' + failed + ' failed' : '') + '. Reload to check, and run again if any are left.');
    $('us_x').textContent = 'Close';
    $('us_x').onclick = () => box.remove();
  };
})();
