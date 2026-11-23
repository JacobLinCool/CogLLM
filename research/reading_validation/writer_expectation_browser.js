async page => {
  const base = new URL(page.url());
  if (!["127.0.0.1", "localhost"].includes(base.hostname)) throw new Error("Use the local verified L4 replay.");
  const result = [];
  for (const [theme, width] of [["light", 1280], ["dark", 390]]) {
    await page.setViewportSize({width, height: 900});
    base.searchParams.set("__theme", theme);
    await page.goto(base.href);
    await page.getByRole("checkbox", {name: /Show where readers.*next guesses change/}).check();
    await page.getByRole("button", {name: "Load example"}).click();
    const input = page.getByRole("textbox", {name: "English draft"});
    await page.waitForFunction(() => [...document.querySelectorAll("textarea")].some(el => el.value.includes("middle of dinner")));
    const original = await input.inputValue();
    await page.getByRole("button", {name: "Review draft"}).click();
    const draft = page.locator(".writing-review .draft");
    const graph = page.getByRole("img", {name: /two readers give different next-word answers/});
    await graph.waitFor({state: "visible"});
    if (await draft.textContent() !== original) throw new Error("Printed article changed");
    function coordinates(data) {
      return [...data.matchAll(/([ML])([\d.]+) ([\d.]+)/g)].map(m => [m[1], +m[2], +m[3]]);
    }
    const points = coordinates(await graph.locator(".expectation-lower").getAttribute("d"));
    const upper = coordinates(await graph.locator(".expectation-upper").getAttribute("d"));
    if (!await graph.locator(".expectation-band").isVisible() || upper.length !== points.length ||
        upper.some((p,i) => p[1] !== points[i][1] || p[2] > points[i][2])) throw new Error("Invalid uncertainty band.");
    const marks = {};
    for (const direction of ["increase", "decrease"]) {
      marks[direction] = await draft.locator(".expectation-" + direction).evaluateAll(nodes => nodes.map(n => {
        const r = document.createRange();
        r.selectNodeContents(n.closest(".draft"));
        r.setEndBefore(n);
        const start = r.toString().length;
        return {start, end: start+n.textContent.length, text: n.textContent, style: getComputedStyle(n).borderBottomStyle};
      }));
    }
    const requests = [];
    const observe = r => { if (/\/queue|\/call\//.test(r.url())) requests.push(r.url()); };
    page.on("request", observe);
    try {
      await page.getByText("Values at each word", {exact: true}).click();
      const rows = await page.locator(".expectation-table tbody tr").evaluateAll(nodes => nodes.map(row => [...row.cells].map(c => c.textContent)));
      const fit = await graph.boundingBox();
      if (rows.length !== points.length || points.length < 2 || !fit || fit.width > width || fit.width < 210 ||
          points.some(([cmd,x,y],i) => (i === 0 && cmd !== "M") || y < 1 || y > 99 || (i && points[i-1][1] >= x)) ||
          marks.increase.some(m => m.style !== "solid") || marks.decrease.some(m => m.style !== "dotted") ||
          !marks.increase.length || !marks.decrease.length || await page.evaluate(() => document.documentElement.scrollWidth > innerWidth))
        throw new Error("Visible marks, probabilities or responsive layout disagree.");
      await page.getByText("Values at each word", {exact: true}).click();
      if (requests.length) throw new Error("Inspecting word values requested another prediction.");
      result.push({theme, width, sourceMatches: true, requestsWhileInspecting: requests.length, marks, rows, points, upper});
    } finally {
      page.off("request", observe);
    }
  }
  return result;
}
