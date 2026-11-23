async page => {
  const base = new URL(page.url());
  if (!["127.0.0.1", "localhost"].includes(base.hostname)) throw new Error("Verify the original local L4 replay.");
  const fixtures = [
    {id: "1", text: "The city will replace the footbridge after inspectors found cracks in its supports. A shuttle will carry pedestrians across the river during the repairs."},
    {id: "2", text: "Choose a feeder that is appropriate for your yard. Choose a feeder with different types of seeds. Choose a feeder with a special suet feeder. Install your feeder in a safe location."},
    {id: "3", text: "The news you've heard or read when you've heard or read, or read, a few things you've heard."},
  ];
  const evidence = [];
  for (const [theme, width] of [["light", 1280], ["dark", 390]]) {
    await page.setViewportSize({width, height: 900});
    base.searchParams.set("__theme", theme);
    await page.goto(base.toString());
    const mode = page.getByRole("checkbox", {name: /Check whether this summary is understandable/});
    if (await mode.isChecked()) throw new Error("Summary mode is active before selection.");
    await mode.check();
    await page.getByRole("button", {name: "Load example"}).click();
    const input = page.getByRole("textbox", {name: "English draft"});
    await page.waitForFunction(text => [...document.querySelectorAll("textarea")].some(e => e.value === text), fixtures[0].text);
    for (const fixture of fixtures) {
      if (fixture.id !== "1") await input.fill(fixture.text);
      await page.getByRole("button", {name: "Review draft"}).click();
      await page.waitForFunction(text => document.querySelector(".writing-review .draft")?.textContent === text, fixture.text);
      const scope = page.locator(".writing-review .summary-understanding");
      await scope.getByRole("heading", {name: "Can a reader understand this summary?"}).waitFor({state:"visible"});
      const values = await scope.evaluate(e => {
        const rect = e.getBoundingClientRect();
        const style=getComputedStyle(e.querySelector("strong"));
        return {
          decision: e.querySelector("p").textContent,
          percent: e.querySelector("strong").textContent,
          caption: e.querySelector(".reading-caption").textContent,
          foreground: style.color,
          fits: rect.width >= 210 && rect.width <= innerWidth && document.documentElement.scrollWidth <= innerWidth,
        };
      });
      if (!values.fits || !values.caption.includes("whole draft")) throw new Error("Reading scope or responsive width was lost.");
      if ((theme === "light" && fixture.id === "3") || (theme === "dark" && fixture.id === "1")) {
        await page.locator(".writing-review").screenshot({path: "output/playwright/summary-" + theme + ".png"});
      }
      evidence.push({theme, width, id: fixture.id, text: await input.inputValue(), ...values});
    }
  }
  return evidence;
}
