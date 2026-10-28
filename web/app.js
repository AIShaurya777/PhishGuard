// PhishGuard project site: the link checker and the charts in "What we found".
// Every number comes from the server (/api/numbers, /api/check); nothing is typed in here.

const form = document.querySelector("#check-form");
const input = document.querySelector("#url");
const button = document.querySelector("#check-button");
const errorBox = document.querySelector("#error");
const resultBox = document.querySelector("#result");

const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "style") {
      for (const [property, setting] of Object.entries(value)) node.style.setProperty(property.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`), setting);
    }
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child != null) node.append(child);
  }
  return node;
}

// ---------- Link checker ----------

async function check(url) {
  errorBox.hidden = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = "Checking…";
  try {
    const response = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "That link couldn’t be checked.");
    showResult(body);
  } catch (error) {
    resultBox.hidden = true;
    errorBox.textContent = error instanceof TypeError
      ? "The PhishGuard server isn’t running. Start it with: .venv/bin/python -m phishguard.web"
      : error.message;
    errorBox.hidden = false;
  } finally {
    button.removeAttribute("aria-busy");
    button.textContent = "Check link";
  }
}

function specimen(result) {
  const scores = result.char_scores;
  const largest = Math.max(...scores.map(Math.abs), 1e-6);
  const line = el("p", { class: "specimen is-hidden", "aria-label": `Link: ${result.url}. The model reads ${result.domain}.` });
  for (const part of result.parts) {
    if (!part.read) {
      line.append(el("span", { class: "ignored", "aria-hidden": "true" }, part.text));
      continue;
    }
    const read = el("span", { class: "read", "aria-hidden": "true" });
    [...part.text].forEach((char, i) => {
      const score = scores[i] ?? 0;
      read.append(el("span", {
        class: "ch",
        title: `${char}: ${score >= 0 ? "+" : ""}${score.toFixed(2)}`,
        style: {
          "--tone": score >= 0 ? "var(--phish)" : "var(--legit)",
          "--weight": (0.06 + 0.6 * Math.abs(score) / largest).toFixed(3),
          "--i": i,
        },
      }, char));
    });
    line.append(read);
  }
  // One reveal: the shading fills in left to right once the result is on screen.
  requestAnimationFrame(() => requestAnimationFrame(() => line.classList.remove("is-hidden")));
  return line;
}

function verdict(name, model) {
  const call = model.phishing ? "Looks like phishing" : "Looks legitimate";
  const tone = model.phishing ? "is-phishing" : "is-legit";
  return el("div", { class: "verdict" },
    el("h3", {}, name),
    el("p", { class: `verdict-call ${tone}` }, call),
    el("div", { class: "meter", role: "img", "aria-label": `Phishing score ${pct(model.probability)}, flagged above ${pct(model.threshold)}` },
      el("span", { class: `meter-fill ${tone}`, style: { width: pct(model.probability, 1) } }),
      el("span", { class: "meter-threshold", style: { left: pct(model.threshold, 1) } })),
    el("p", { class: "verdict-detail" },
      `Phishing score ${pct(model.probability, 1)}. It calls a link phishing above ${pct(model.threshold)}, the line on the bar.`));
}

function drivers(features) {
  const largest = Math.max(...features.map((f) => Math.abs(f.score)), 1e-6);
  return el("ul", { class: "drivers" }, features.map((f) => {
    const direction = f.score >= 0 ? "towards phishing" : "towards legitimate";
    const width = `${(50 * Math.abs(f.score) / largest).toFixed(1)}%`;
    return el("li", { class: "driver" },
      el("span", {}, `${f.name}: `, el("span", { class: "driver-value" }, String(f.value))),
      el("span", { class: "driver-bar", role: "img", "aria-label": `${direction}, ${Math.abs(f.score).toFixed(2)}` },
        el("span", { class: f.score >= 0 ? "towards-phish" : "towards-legit", style: { width } })));
  }));
}

function showResult(result) {
  const warnings = result.warnings.length
    ? el("ul", { class: "warnings" }, result.warnings.map((w) => el("li", {}, w)))
    : el("p", { class: "no-warnings" }, "None of our known weak spots apply to this link. That still doesn’t make it safe.");

  resultBox.replaceChildren(
    specimen(result),
    el("p", { class: "specimen-key" },
      "Crossed-out parts are ignored. ",
      el("span", { class: "swatch phish" }), " Red letters pushed the hybrid model towards phishing, ",
      el("span", { class: "swatch legit" }), " blue towards legitimate. Darker means a bigger push."),
    el("div", { class: "verdicts" },
      verdict("Hybrid model", result.models.hybrid),
      verdict("Character n-gram model", result.models.ngram)),
    el("div", { class: "result-columns" },
      el("div", {},
        el("h3", {}, "Read this result with care"),
        el("p", {}, "Known weak spots that apply to this link, from our tests."),
        warnings),
      el("div", {},
        el("h3", {}, "What moved the hybrid model"),
        el("p", {}, "Measured properties of the domain, biggest effect first."),
        drivers(result.features))),
  );
  resultBox.hidden = false;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  check(input.value);
});

document.querySelectorAll(".examples button").forEach((example) => {
  example.addEventListener("click", () => {
    input.value = example.dataset.url;
    check(example.dataset.url);
  });
});

// ---------- Charts in "What we found" ----------

function barRow(label, value, { min = 0, tone = "", text }) {
  const width = Math.max(0, (value - min) / (1 - min));
  return el("div", { class: "chart-row" },
    el("span", { class: "chart-label" }, label),
    el("span", { class: "chart-track", "aria-hidden": "true" },
      el("span", { class: `chart-bar ${tone}`, style: { width: pct(width, 1) } })),
    el("span", { class: "chart-value" }, text));
}

function axis(left, right) {
  return el("div", { class: "chart-axis", "aria-hidden": "true" },
    el("span"), el("span", {}, el("span", {}, left), el("span", {}, right)), el("span"));
}

function accuracyChart(rows) {
  const figure = document.querySelector("#accuracy-chart");
  const groups = [
    ["full link", "Full link: the shortcuts are visible"],
    ["domain only", "Domain only: what this site uses"],
  ];
  figure.replaceChildren(
    ...groups.map(([input, title]) => el("div", { class: "chart-group" },
      el("h4", {}, title),
      rows.filter((r) => r.input === input).map((r) => barRow(r.model, r.accuracy, {
        min: 0.5,
        tone: input === "domain only" ? "is-honest" : "",
        text: el("span", {}, pct(r.accuracy, 1), " ", el("small", {}, `± ${pct(r.sd, r.sd < 0.001 ? 2 : 1)}`)),
      })))),
    axis("50%", "100%"),
    el("figcaption", {}, "Test accuracy, mean ± standard deviation over five splits. Bars start at 50%; always guessing “legitimate” would score 58%."),
  );
  figure.removeAttribute("aria-busy");
}

function realSitesChart(numbers) {
  const figure = document.querySelector("#real-sites-chart");
  const sets = [
    ["CrUX top 10k: bare domain", "Chrome’s most visited sites, plain domain", "like github.com"],
    ["CrUX top 10k: with a subdomain", "Chrome’s most visited sites with a subdomain", "like mail.yahoo.com"],
    ["Tranco top 20k: registered domains", "Tranco research ranking, top 20,000", "plain domains"],
  ];
  const models = [["hybrid", "Hybrid"], ["ngram", "Character n-grams"]];
  figure.replaceChildren(
    ...sets.map(([key, title, example]) => {
      const set = numbers.real_sites[key];
      return el("div", { class: "chart-group" },
        el("h4", {}, `${title} (${set.sites.toLocaleString()} sites, ${example})`),
        models.map(([m, label]) => barRow(label, set[m], { tone: "is-alarm", text: `${pct(set[m])} flagged` })));
    }),
    axis("0%", "100%"),
    el("figcaption", {}, "Share of real, popular websites wrongly flagged as phishing. None of these sites are in the dataset. Ideally every bar would be empty."),
  );
  figure.removeAttribute("aria-busy");
}

function attacksTable(numbers) {
  const https = numbers.raw_url_attacks["switch to HTTPS"];
  const www = numbers.raw_url_attacks["https://www. + no path"];
  const edits = numbers.all_edits;
  const mail = numbers.legit_with_subdomain;
  const row = (what, model, before, after, label) => el("tr", {},
    el("th", { scope: "row" }, what), el("td", {}, model),
    el("td", { class: "num" }, pct(before)), el("td", { class: "num drop" }, pct(after)), el("td", {}, label));

  document.querySelector("#attacks-table").replaceChildren(el("table", { class: "attacks" },
    el("caption", {}, "Cheap changes, before and after"),
    el("thead", {}, el("tr", {},
      el("th", { scope: "col" }, "Change"), el("th", { scope: "col" }, "Model"),
      el("th", { scope: "col" }, "Before"), el("th", { scope: "col" }, "After"), el("th", { scope: "col" }, "Measured"))),
    el("tbody", {},
      row("Phishing link switched from http to https", "Full-link n-grams", https.before, https.after, "phishing caught"),
      row("Phishing link rewritten as https://www.<domain>", "Full-link n-grams", www.before, www.after, "phishing caught"),
      row("Phishing domain given every cheap edit: own .com, no subdomain, no digits or hyphens", "Hybrid", edits.hybrid.before, edits.hybrid.after, "phishing caught"),
      row("Same edits", "Character n-grams", edits.ngram.before, edits.ngram.after, "phishing caught"),
      row("Legitimate domain given a mail. subdomain", "Hybrid", mail.original.hybrid, mail.mail.hybrid, "false alarms"),
    )));
}

fetch("/api/numbers")
  .then((response) => response.json())
  .then((numbers) => {
    accuracyChart(numbers.accuracy);
    realSitesChart(numbers);
    attacksTable(numbers);
  })
  .catch(() => {
    document.querySelectorAll(".chart .loading").forEach((p) => {
      p.textContent = "Results couldn’t load. Start the server with: .venv/bin/python -m phishguard.web";
    });
  });

// A link in the address bar (?url=...) is checked straight away, so a result can be shared.
const linked = new URLSearchParams(location.search).get("url");
if (linked) {
  input.value = linked;
  check(linked);
}
