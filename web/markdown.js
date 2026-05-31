/* Minimal, dependency-free Markdown -> HTML renderer.
 * Covers what the blog engine emits: headings, fenced code, inline code,
 * bold/italic, links, ordered/unordered lists, blockquotes, tables, rules.
 * All text is HTML-escaped first, so model output cannot inject markup. */
(function () {
  "use strict";

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Inline formatting on already-escaped text.
  function inline(text) {
    return text
      .split(/(`[^`]+`)/g)
      .map(function (part) {
        if (part.length >= 2 && part[0] === "`" && part[part.length - 1] === "`") {
          return "<code>" + part.slice(1, -1) + "</code>";
        }
        return part
          .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener">$1</a>')
          .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
          .replace(/__([^_]+)__/g, "<strong>$1</strong>")
          .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, "$1<em>$2</em>")
          .replace(/(^|[^_\w])_([^_\s][^_]*?)_/g, "$1<em>$2</em>");
      })
      .join("");
  }

  function isTableSeparator(line) {
    return /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(line) && line.indexOf("-") !== -1;
  }

  function splitRow(line) {
    var cells = line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|");
    return cells.map(function (c) { return c.trim(); });
  }

  function renderMarkdown(md) {
    if (!md) return "";
    md = md.replace(/\r\n/g, "\n");

    // 1) Pull out fenced code blocks so their contents are not parsed.
    var codeBlocks = [];
    md = md.replace(/```([\w+-]*)\n([\s\S]*?)```/g, function (_, lang, code) {
      var token = "@@CODEBLOCK" + codeBlocks.length + "@@";
      var label = lang ? '<span class="code-lang">' + escapeHtml(lang) + "</span>" : "";
      codeBlocks.push("<pre>" + label + "<code>" + escapeHtml(code.replace(/\n$/, "")) + "</code></pre>");
      return "\n" + token + "\n";
    });

    var lines = md.split("\n");
    var html = [];
    var i = 0;
    var para = [];
    var list = null; // {type: 'ul'|'ol', items: []}

    function flushPara() {
      if (para.length) { html.push("<p>" + inline(escapeHtml(para.join(" "))) + "</p>"); para = []; }
    }
    function flushList() {
      if (list) {
        html.push("<" + list.type + ">" + list.items.map(function (it) {
          return "<li>" + inline(escapeHtml(it)) + "</li>";
        }).join("") + "</" + list.type + ">");
        list = null;
      }
    }
    function flushAll() { flushPara(); flushList(); }

    while (i < lines.length) {
      var line = lines[i];
      var trimmed = line.trim();

      if (trimmed === "") { flushAll(); i++; continue; }

      var codeMatch = trimmed.match(/^@@CODEBLOCK(\d+)@@$/);
      if (codeMatch) { flushAll(); html.push(codeBlocks[+codeMatch[1]]); i++; continue; }

      var heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (heading) { flushAll(); var lvl = heading[1].length; html.push("<h" + lvl + ">" + inline(escapeHtml(heading[2])) + "</h" + lvl + ">"); i++; continue; }

      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) { flushAll(); html.push("<hr>"); i++; continue; }

      // Table: a row with a pipe followed by a separator row.
      if (trimmed.indexOf("|") !== -1 && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
        flushAll();
        var headers = splitRow(trimmed);
        var rows = [];
        i += 2;
        while (i < lines.length && lines[i].trim().indexOf("|") !== -1 && lines[i].trim() !== "") {
          rows.push(splitRow(lines[i])); i++;
        }
        html.push("<table><thead><tr>" + headers.map(function (h) { return "<th>" + inline(escapeHtml(h)) + "</th>"; }).join("") + "</tr></thead><tbody>" +
          rows.map(function (r) { return "<tr>" + r.map(function (c) { return "<td>" + inline(escapeHtml(c)) + "</td>"; }).join("") + "</tr>"; }).join("") +
          "</tbody></table>");
        continue;
      }

      var ulMatch = trimmed.match(/^[-*+]\s+(.*)$/);
      var olMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (ulMatch || olMatch) {
        flushPara();
        var type = ulMatch ? "ul" : "ol";
        if (!list || list.type !== type) { flushList(); list = { type: type, items: [] }; }
        list.items.push((ulMatch ? ulMatch[1] : olMatch[1]));
        i++; continue;
      }

      if (trimmed[0] === ">") {
        flushAll();
        var quote = [];
        while (i < lines.length && lines[i].trim()[0] === ">") {
          quote.push(lines[i].trim().replace(/^>\s?/, "")); i++;
        }
        html.push("<blockquote>" + inline(escapeHtml(quote.join(" "))) + "</blockquote>");
        continue;
      }

      // default: paragraph text
      flushList();
      para.push(trimmed);
      i++;
    }
    flushAll();
    return html.join("\n");
  }

  window.renderMarkdown = renderMarkdown;
})();
