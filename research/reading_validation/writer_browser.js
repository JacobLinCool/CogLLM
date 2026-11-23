async page => {
  const url = new URL(page.url());
  if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
    throw new Error("A locally served replay of the real L4 outputs is required.");
  }
  const observations = [];
  for (const {theme, width, narrative, terminology, references = false} of [
    {theme: "light", width: 1280, narrative: false, terminology: false},
    {theme: "dark", width: 390, narrative: true, terminology: false},
    {theme: "light", width: 1280, narrative: false, terminology: true},
    {theme: "dark", width: 390, narrative: false, terminology: true},
    {theme: "light", width: 1280, narrative: false, terminology: false, references: true},
    {theme: "dark", width: 390, narrative: true, terminology: false, references: true},
  ]) {
    await page.setViewportSize({width, height: 900});
    url.searchParams.set("__theme", theme);
    await page.goto(url.toString());
    const checkbox = page.getByRole("checkbox", {name: /Check character introductions/});
    await checkbox.setChecked(narrative);
    await page.getByRole("checkbox", {name: /Track technical terms/}).setChecked(terminology);
    await page.getByRole("checkbox", {name: /Inspect a pronoun that refers back/}).setChecked(references);
    await page.getByRole("button", {name: "Load example"}).click();
    await page.waitForFunction(() => Array.from(document.querySelectorAll("textarea")).some(box => box.value.length > 0));
    await page.getByRole("button", {name: "Review draft"}).click();
    const draft = page.locator(".writing-review .draft");
    await draft.waitFor({state: "visible"});
    const text = await page.getByRole("textbox", {name: "English draft"}).inputValue();
    if (await draft.textContent() !== text || !text.length) {
      throw new Error("The article view differs from the original draft.");
    }
    const geometry = await draft.evaluate(element => {
      function luminance(color) {
        const values = color.match(/[\d.]+/g).slice(0, 3).map(channel => {
          const value = Number(channel) / 255;
          return value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4;
        });
        return .2126 * values[0] + .7152 * values[1] + .0722 * values[2];
      }
      const marks = Array.from(element.querySelectorAll("mark"));
      const contrasts = marks.map(mark => {
        const style = getComputedStyle(mark);
        const l1 = luminance(style.color);
        const l2 = luminance(style.backgroundColor);
        return (Math.max(l1,l2) + .05) / (Math.min(l1,l2) + .05);
      });
      return {
        sentences: marks.length,
        wordMarks: element.querySelectorAll(".phrase").length,
        characterMarks: element.querySelectorAll(".character").length,
        lowestContrast: Math.min(...contrasts),
        viewportWidth: document.documentElement.clientWidth,
        documentWidth: document.documentElement.scrollWidth,
      };
    });
    if (!geometry.sentences || !geometry.wordMarks || geometry.lowestContrast < 4.5
      || geometry.documentWidth !== geometry.viewportWidth || (narrative && !geometry.characterMarks)) {
      throw new Error("Article marks, contrast, or mobile fit failed: " + JSON.stringify(geometry));
    }
    if (terminology) {
      const graph = page.getByRole("img", {name: /Estimated mentions awaiting a definition/});
      await graph.waitFor({state: "visible"});
      const view = await graph.locator("svg").boundingBox();
      const sourceCount = await draft.locator(".term-source").count();
      const cueCount = await draft.locator(".term-cue").count();
      const points = await graph.locator("polyline").getAttribute("points");
      if (!view || view.width < 180 || view.height < 30 || !sourceCount || !cueCount || points.split(" ").length !== 46) {
        throw new Error("The expected-count curve or its named term links are missing.");
      }
      Object.assign(geometry, {sourceMarks: sourceCount, cueMarks: cueCount, curvePoints: points.split(" ").length});
    }
    if (references) {
      const choices = page.locator(".ref-review details");
      const offered = choices.filter({has: page.locator(".ref-answer strong")});
      if (!await offered.count() || await choices.locator(":scope[open]").count()) {
        throw new Error("An inspectable earlier phrase must be available on request.");
      }
      const requestUrls = [];
      const observe = req => {if (/\/queue|\/call\//.test(req.url())) requestUrls.push(req.url());};
      page.on("request", observe);
      try {
        const chosen = offered.first();
        await chosen.locator("summary").click();
        const className = await chosen.getAttribute("class");
        const earlier = draft.locator("." + className.replace("ref-", "ref-earlier-"));
        const pronoun = draft.locator("." + className.replace("ref-", "ref-pronoun-"));
        const phrase = (await chosen.locator(".ref-answer strong").textContent()).slice(1, -1);
        if (!phrase || (await earlier.allTextContents()).join("") !== phrase
            || await earlier.first().evaluate(el => getComputedStyle(el).boxShadow) === "none"
            || await pronoun.first().evaluate(el => getComputedStyle(el).outlineStyle) === "none"
            || await draft.textContent() !== text) {
          throw new Error("The selected pronoun and suggested phrase must remain visible in the original draft.");
        }
        await chosen.locator("summary").click();
        if (requestUrls.length || await choices.evaluateAll(nodes => nodes.some(el => el.open))) {
          throw new Error("Inspecting a cached reference unexpectedly invoked the model or left the choice open.");
        }
        Object.assign(geometry, {pronounChoices: await choices.count(), offeredLinks: await offered.count(), highlightedPhrase: phrase});
      } finally {
        page.off("request", observe);
      }
    }
    observations.push({theme, width, narrative, terminology, references, characters: text.length, sourceTextMatches: true, ...geometry});
  }
  return observations;
}
