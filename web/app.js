/* Blog Writing Agent — frontend logic (vanilla JS, no build step). */
(function () {
  "use strict";

  var view = document.getElementById("view");
  var META = null;
  var activeStream = null; // EventSource
  var activeTimer = null;   // setInterval id
  var activePoll = null;    // fallback poll interval

  var NODE_LABELS = {
    router: "Routing & research decision",
    audience_adapter: "Audience adaptation",
    research: "Web research",
    citation_enricher: "Citation enrichment",
    orchestrator: "Planning the outline",
    persona: "Persona debate",
    worker_pipeline: "Writing sections",
    reducer: "Merging sections",
    humanizer: "Humanizing",
    seo_optimizer: "SEO metadata",
    image_pipeline: "Images",
    quality_scoring: "Quality scoring",
    export: "Exporting artifacts",
    dashboard: "Telemetry & dashboard",
  };

  // ---- utilities ----------------------------------------------------------

  function api(path, opts) {
    return fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}))
      .then(function (r) {
        if (!r.ok) return r.json().catch(function () { return {}; }).then(function (b) { throw new Error(b.detail || ("HTTP " + r.status)); });
        return r.status === 204 ? null : r.json();
      });
  }

  function el(html) { var t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; }
  function tpl(id) { return document.getElementById(id).content.cloneNode(true); }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function toast(msg) {
    var t = el('<div class="toast">' + msg + "</div>");
    document.body.appendChild(t);
    requestAnimationFrame(function () { t.classList.add("show"); });
    setTimeout(function () { t.classList.remove("show"); setTimeout(function () { t.remove(); }, 300); }, 2200);
  }

  function fmtElapsed(ms) {
    var s = Math.max(0, Math.floor(ms / 1000));
    var m = Math.floor(s / 60); s = s % 60;
    var h = Math.floor(m / 60); m = m % 60;
    var mm = (h > 0 ? String(m).padStart(2, "0") : m) + ":" + String(s).padStart(2, "0");
    return h > 0 ? h + ":" + mm : mm;
  }

  function fmtDate(iso) {
    if (!iso) return "—";
    try { return new Date(iso).toLocaleString(); } catch (e) { return iso; }
  }

  function cleanupActive() {
    if (activeStream) { activeStream.close(); activeStream = null; }
    if (activeTimer) { clearInterval(activeTimer); activeTimer = null; }
    if (activePoll) { clearInterval(activePoll); activePoll = null; }
  }

  // ---- health -------------------------------------------------------------

  function pollHealth() {
    api("/api/health").then(function (h) {
      document.getElementById("health-dot").className = "health-dot ok";
      document.getElementById("health-text").textContent = h.queue_depth ? (h.queue_depth + " queued") : "ready";
    }).catch(function () {
      document.getElementById("health-dot").className = "health-dot down";
      document.getElementById("health-text").textContent = "offline";
    });
  }

  // ---- views --------------------------------------------------------------

  function renderNew() {
    cleanupActive();
    clear(view); view.appendChild(tpl("tpl-new"));

    function fill(id, values, selected) {
      var sel = document.getElementById(id);
      values.forEach(function (v) {
        var o = document.createElement("option");
        o.value = v; o.textContent = v.charAt(0).toUpperCase() + v.slice(1);
        if (v === selected) o.selected = true;
        sel.appendChild(o);
      });
    }
    fill("sel-audience", META.audience_modes, "engineer");
    fill("sel-execution", META.execution_modes, "balanced");
    fill("sel-image", META.image_modes, "off");

    var form = document.getElementById("blog-form");
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var btn = document.getElementById("submit-btn");
      var err = document.getElementById("form-error");
      err.textContent = "";
      var data = {
        topic: form.topic.value.trim(),
        audience_mode: form.audience_mode.value,
        execution_mode: form.execution_mode.value,
        image_mode: form.image_mode.value,
        as_of: form.as_of.value || null,
      };
      if (data.topic.length < 4) { err.textContent = "Please enter a more descriptive topic."; return; }
      btn.disabled = true; btn.textContent = "Starting…";
      api("/api/blogs", { method: "POST", body: JSON.stringify(data) })
        .then(function (job) { location.hash = "#/job/" + job.id; })
        .catch(function (e2) { err.textContent = e2.message; btn.disabled = false; btn.textContent = "Generate blog"; });
    });
  }

  function renderTimeline(container, progress, status) {
    var completed = (progress && progress.completed_nodes) || [];
    var current = progress && progress.current_node;
    var lastIdx = -1;
    META.pipeline_nodes.forEach(function (n, idx) { if (completed.indexOf(n) !== -1) lastIdx = Math.max(lastIdx, idx); });

    clear(container);
    META.pipeline_nodes.forEach(function (node, idx) {
      var state = "pending", icon = idx + 1;
      if (completed.indexOf(node) !== -1) { state = "done"; icon = "✓"; }
      else if (node === current && status === "running") { state = "current"; icon = "•"; }
      else if (idx < lastIdx) { state = "skipped"; icon = "–"; } // e.g. research not needed
      var li = el('<li class="' + (state === "skipped" ? "" : state) + '"><span class="tl-icon">' + icon + "</span><span>" + (NODE_LABELS[node] || node) + (state === "skipped" ? " (skipped)" : "") + "</span></li>");
      container.appendChild(li);
    });
  }

  function renderGeneration(jobId) {
    cleanupActive();
    clear(view); view.appendChild(tpl("tpl-generation"));
    var done = false;

    function paint(job) {
      document.getElementById("gen-topic").textContent = job.topic || "…";
      var badge = document.getElementById("gen-status");
      badge.textContent = job.status; badge.className = "badge " + job.status;
      var prog = job.progress || {};
      document.getElementById("gen-fill").style.width = (prog.percent || 0) + "%";
      document.getElementById("gen-percent").textContent = prog.percent || 0;
      document.getElementById("gen-current").textContent =
        job.status === "running" ? (NODE_LABELS[prog.current_node] || "working…") :
        job.status === "queued" ? "waiting in queue…" : job.status;
      renderTimeline(document.getElementById("gen-timeline"), prog, job.status);
      if (job.error) document.getElementById("gen-error").textContent = job.error;
    }

    function startTimer(startedAt) {
      var base = startedAt ? new Date(startedAt).getTime() : Date.now();
      activeTimer = setInterval(function () {
        document.getElementById("gen-timer").textContent = fmtElapsed(Date.now() - base);
      }, 1000);
    }

    function onTerminal(status) {
      if (done) return; done = true;
      cleanupActive();
      if (status === "succeeded") { toast("Blog ready"); renderResult(jobId); }
      else { api("/api/blogs/" + jobId).then(paint); } // show failed/cancelled state
    }

    document.getElementById("cancel-btn").addEventListener("click", function () {
      this.disabled = true; this.textContent = "Cancelling…";
      api("/api/blogs/" + jobId + "/cancel", { method: "POST" }).catch(function (e) { toast(e.message); });
    });

    // Initial load, then stream.
    api("/api/blogs/" + jobId).then(function (job) {
      paint(job);
      startTimer(job.started_at);
      if (["succeeded", "failed", "cancelled"].indexOf(job.status) !== -1) { onTerminal(job.status); return; }
      subscribe();
    }).catch(function (e) {
      document.getElementById("gen-error").textContent = e.message;
    });

    function subscribe() {
      try {
        activeStream = new EventSource("/api/blogs/" + jobId + "/events");
        activeStream.onmessage = function (ev) {
          var d = JSON.parse(ev.data);
          paint({ topic: document.getElementById("gen-topic").textContent, status: d.status, progress: d.progress, error: d.error });
          if (["succeeded", "failed", "cancelled"].indexOf(d.status) !== -1) onTerminal(d.status);
        };
        activeStream.addEventListener("done", function () { });
        activeStream.onerror = function () { if (!done) startPollFallback(); };
      } catch (e) { startPollFallback(); }
    }

    function startPollFallback() {
      if (activePoll || done) return;
      if (activeStream) { activeStream.close(); activeStream = null; }
      activePoll = setInterval(function () {
        api("/api/blogs/" + jobId).then(function (job) {
          paint(job);
          if (["succeeded", "failed", "cancelled"].indexOf(job.status) !== -1) onTerminal(job.status);
        }).catch(function () { });
      }, 2500);
    }
  }

  function renderResult(jobId) {
    cleanupActive();
    api("/api/blogs/" + jobId + "/result").then(function (res) {
      clear(view); view.appendChild(tpl("tpl-result"));
      document.getElementById("res-topic").textContent = res.topic;

      var q = res.quality_score || {};
      var stats = document.getElementById("res-stats");
      stats.appendChild(el('<span class="stat"><b>' + res.word_count + "</b> words</span>"));
      if (q.overall) stats.appendChild(el('<span class="stat">Quality <b>' + q.overall + "/10</b></span>"));
      stats.appendChild(el('<span class="stat">run <b>' + res.run_id + "</b></span>"));

      // Blog
      document.getElementById("res-markdown").innerHTML = window.renderMarkdown(res.markdown);

      // SEO
      var seo = res.seo_metadata || {};
      var seoBox = document.getElementById("res-seo");
      function kv(k, v) { return '<div class="kv"><div class="k">' + k + '</div><div class="v">' + (v || "—") + "</div></div>"; }
      seoBox.innerHTML =
        kv("Meta title", esc(seo.meta_title)) +
        kv("Meta description", esc(seo.meta_description)) +
        kv("Slug", seo.slug ? "<code>" + esc(seo.slug) + "</code>" : "") +
        '<div class="kv"><div class="k">Keywords</div><div class="v chips">' +
          ((seo.keywords || []).map(function (k) { return '<span class="chip">' + esc(k) + "</span>"; }).join("") || "—") + "</div></div>" +
        (seo.faq_block ? '<div class="kv"><div class="k">FAQ</div><div class="v markdown">' + window.renderMarkdown(seo.faq_block) + "</div></div>" : "");

      // Quality
      var qBox = document.getElementById("res-quality");
      var dims = [["clarity", "Clarity"], ["technical_depth", "Technical depth"], ["seo_readiness", "SEO readiness"], ["redundancy", "Low redundancy"], ["hallucination_risk", "Hallucination risk"], ["overall", "Overall"]];
      if (Object.keys(q).length) {
        qBox.innerHTML = '<div class="score-grid">' + dims.filter(function (d) { return q[d[0]] != null; }).map(function (d) {
          var val = q[d[0]];
          return '<div class="score-row"><div class="score-top"><span>' + d[1] + '</span><b>' + val + '/10</b></div><div class="meter"><span style="width:' + (val * 10) + '%"></span></div></div>';
        }).join("") + "</div>" + (q.summary ? '<div class="score-summary">' + esc(q.summary) + "</div>" : "");
      } else {
        qBox.innerHTML = '<p class="muted">No quality score recorded for this run.</p>';
      }

      // Images
      var imgBox = document.getElementById("res-images");
      if (res.images && res.images.length) {
        imgBox.innerHTML = res.images.map(function (src) { return '<figure><img src="' + src + '" loading="lazy" alt="figure"></figure>'; }).join("");
      } else {
        imgBox.innerHTML = '<p class="muted">This run is text-only (no images).</p>';
      }

      // Tabs
      view.querySelectorAll(".tab").forEach(function (tab) {
        tab.addEventListener("click", function () {
          view.querySelectorAll(".tab").forEach(function (t) { t.classList.remove("is-active"); });
          view.querySelectorAll(".tab-panel").forEach(function (p) { p.classList.remove("is-active"); });
          tab.classList.add("is-active");
          view.querySelector('.tab-panel[data-panel="' + tab.dataset.tab + '"]').classList.add("is-active");
        });
      });

      // Actions
      var dash = document.getElementById("dashboard-link");
      dash.href = "/api/runs/" + res.run_id + "/dashboard";
      document.getElementById("download-btn").addEventListener("click", function () {
        var blob = new Blob([res.markdown], { type: "text/markdown" });
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = (res.topic || "blog").replace(/[^\w\- ]+/g, "").slice(0, 60).trim() + ".md";
        a.click(); URL.revokeObjectURL(a.href);
      });
      document.getElementById("copy-btn").addEventListener("click", function () {
        navigator.clipboard.writeText(res.markdown).then(function () { toast("Markdown copied"); }, function () { toast("Copy failed"); });
      });
    }).catch(function (e) {
      clear(view);
      view.appendChild(el('<div class="card"><h1>Could not load result</h1><p class="muted">' + esc(e.message) + '</p><p><a href="#/history">Back to history</a></p></div>'));
    });
  }

  function esc(s) { return (s == null ? "" : String(s)).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  function renderHistory() {
    cleanupActive();
    clear(view); view.appendChild(tpl("tpl-history"));
    document.getElementById("refresh-history").addEventListener("click", loadHistory);
    loadHistory();

    function loadHistory() {
      api("/api/blogs?limit=100").then(function (jobs) {
        var body = document.getElementById("history-body");
        var empty = document.getElementById("history-empty");
        clear(body);
        if (!jobs.length) { empty.hidden = false; return; }
        empty.hidden = true;
        jobs.forEach(function (j) {
          var tr = el(
            "<tr>" +
            '<td class="topic-cell">' + esc(j.topic) + "</td>" +
            "<td>" + esc(j.audience_mode) + "</td>" +
            '<td><span class="badge ' + j.status + '">' + j.status + "</span></td>" +
            "<td>" + (j.word_count || "—") + "</td>" +
            "<td>" + (j.quality_overall ? j.quality_overall + "/10" : "—") + "</td>" +
            "<td>" + fmtDate(j.created_at) + "</td>" +
            "</tr>");
          tr.addEventListener("click", function () { location.hash = "#/job/" + j.id; });
          body.appendChild(tr);
        });
      }).catch(function (e) { toast(e.message); });
    }
  }

  // ---- router -------------------------------------------------------------

  function setActiveNav(route) {
    document.querySelectorAll(".nav-link").forEach(function (a) {
      a.classList.toggle("is-active", a.dataset.route === route);
    });
  }

  function route() {
    var hash = location.hash || "#/new";
    var parts = hash.replace(/^#\//, "").split("/");
    if (parts[0] === "job" && parts[1]) {
      setActiveNav("history");
      // Decide generation vs result based on status.
      api("/api/blogs/" + parts[1]).then(function (job) {
        if (job.status === "succeeded") renderResult(parts[1]);
        else renderGeneration(parts[1]);
      }).catch(function () {
        clear(view); view.appendChild(el('<div class="card"><h1>Job not found</h1><p><a href="#/history">Back to history</a></p></div>'));
      });
    } else if (parts[0] === "history") {
      setActiveNav("history"); renderHistory();
    } else {
      setActiveNav("new"); renderNew();
    }
  }

  // ---- boot ---------------------------------------------------------------

  window.addEventListener("hashchange", route);

  api("/api/meta").then(function (m) {
    META = m;
    pollHealth(); setInterval(pollHealth, 8000);
    if (!location.hash) location.hash = "#/new";
    route();
  }).catch(function () {
    view.innerHTML = '<div class="card"><h1>Backend unavailable</h1><p class="muted">Could not reach the API. Start it with <code>python -m uvicorn api.main:app --port 8000</code> and reload.</p></div>';
  });
})();
