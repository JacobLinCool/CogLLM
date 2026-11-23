async page => {
  const base = new URL(page.url());
  if (!["127.0.0.1","localhost"].includes(base.hostname)) throw new Error("A localhost replay is required");
  const observed = [];
  for (const [theme,width] of [["light",1280],["dark",390]]) {
    await page.setViewportSize({width,height:900});
    base.searchParams.set("__theme",theme);
    await page.goto(base.href);
    await page.getByText("Do two events in a short story conflict?",{exact:true}).click();
    await page.getByRole("button",{name:"Load a physical story"}).click();
    const box=page.getByRole("textbox",{name:"Short physical story"});
    await page.waitForFunction(()=>[...document.querySelectorAll("textarea")].some(el=>el.value.includes("glass bottle")));
    const text=await box.inputValue();
    await page.getByRole("button",{name:"Compare events"}).click();
    const draft=page.locator(".physical-draft");
    await draft.waitFor({state:"visible"});
    if (await draft.textContent()!==text) throw new Error("Draft differs from the original");
    const marks=await draft.locator("mark").evaluateAll(nodes=>nodes.map(el=>({
      role:el.className,start:+el.dataset.start,end:+el.dataset.end,text:el.textContent,
      background:getComputedStyle(el).backgroundColor,color:getComputedStyle(el).color
    })));
    if (marks.length!==2 || marks[0].role!=="physical-earlier"||marks[1].role!=="physical-conflict"||
        marks.some(m=>text.slice(m.start,m.end)!==m.text)||marks[0].end>marks[1].start||marks[0].background===marks[1].background)
      throw new Error("Both exact and visually distinct source spans must be visible");
    const rect=await draft.boundingBox();
    if (rect.width>width||rect.x<0) throw new Error("Story cannot be read at this width");
    observed.push({theme,width,wholeTextVisible:true,marks});
  }
  return observed;
}
