async page => {
  const url = new URL(page.url());
  if (!["127.0.0.1", "localhost"].includes(url.hostname)) throw new Error("Expected the local signed-output playback.");
  const modes = [
    ["wording", null], ["summary", /Check whether this summary is understandable/],
    ["expectation", /Show where readers.*next guesses change/], ["narrative", /Check character introductions/],
    ["terminology", /Track technical terms/], ["reference", /Inspect a pronoun that refers back/],
    ["physical", null],
  ];
  const audit = root => {
    function rgba(css) {
      const values = css.match(/[\d.]+/g)?.map(Number);
      if (!values || values.length < 3 || values.length > 4 || !css.startsWith("rgb")) throw new Error("Unexpected computed color " + css);
      return [...values.slice(0,3).map(v=>v/255), values[3] ?? 1];
    }
    function over(f,b) {
      const alpha = f[3]+b[3]*(1-f[3]);
      return alpha ? [0,1,2].map(i=>(f[i]*f[3]+b[i]*b[3]*(1-f[3]))/alpha).concat(alpha) : [0,0,0,0];
    }
    function lum(c) {
      return c.slice(0,3).reduce((sum,x,i)=>sum+[.2126,.7152,.0722][i]*(x<=.04045 ? x/12.92 : ((x+.055)/1.055)**2.4),0);
    }
    const roles = {};
    const walker = document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
    let node;
    while(node=walker.nextNode()) {
      const el=node.parentElement;
      if (!node.textContent.trim() || !el || el.closest("style,svg,script") || el.checkVisibility?.()===false) continue;
      const range=document.createRange();range.selectNodeContents(node);
      if (![...range.getClientRects()].some(r=>r.width>0 && r.height>0)) continue;
      let bg=[0,0,0,0], fg=rgba(getComputedStyle(el).color);
      for (let a=el; a; a=a.parentElement) {
        const style=getComputedStyle(a), backdrop=rgba(style.backgroundColor);
        fg=over(fg,backdrop);bg=over(bg,backdrop);
        const opacity=Number(style.opacity);
        fg[3]*=opacity;bg[3]*=opacity;
        if(bg[3]===1 && fg[3]===1) break;
      }
      bg=over(bg,[1,1,1,1]);fg=over(fg,[1,1,1,1]);
      const ratio=(Math.max(lum(fg),lum(bg))+.05)/(Math.min(lum(fg),lum(bg))+.05);
      const role=el.closest("a")?"link":el.closest(".phrase")?"underlined words":el.closest(".reading-key,.reading-caption")?"legend and caption":
        el.closest("summary")?"disclosure":el.closest(".draft,.physical-draft")?"source words":
        el.closest("table")?"table":el.closest("li")?"listed evidence":"explanation";
      const state=roles[role]??{nodes:0,min:Infinity};
      state.nodes++;state.min=Math.min(ratio,state.min);roles[role]=state;
      if(ratio<4.5) throw new Error("Contrast "+ratio.toFixed(2)+" on "+role+" / "+el.tagName+" / "+node.textContent.trim().slice(0,60));
    }
    if(root.matches(".writing-review")){
      const draft=root.querySelector(".draft mark");
      const ink=getComputedStyle(draft).color, darkest="#d78353";
      const rgb=[1,3,5].map(i=>parseInt(darkest.slice(i,i+2),16)/255);
      const a=lum(rgba(ink)), b=lum(rgb);
      const ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
      if(ratio<4.5)throw new Error("Ink fails on the darkest validated sentence color.");
      roles["darkest scale endpoint"]={nodes:1,min:ratio};
    }
    if(!Object.keys(roles).length)throw new Error("The local rendered card had no visible text");
    return roles;
  };
  const observed=[];
  for(const [theme,width] of [["light",1280],["dark",390]]) {
    await page.setViewportSize({width,height:900});url.searchParams.set("__theme",theme);
    for(const [mode,check] of modes) {
      await page.goto(url.href);
      let card;
      if(mode==="physical") {
        await page.getByText("Do two events in a short story conflict?",{exact:true}).click();
        await page.getByRole("button",{name:"Load a physical story"}).click();
        await page.waitForFunction(()=>[...document.querySelectorAll("textarea")].some(e=>e.value.includes("glass bottle")));
        await page.getByRole("button",{name:"Compare events"}).click();card=page.locator(".physical-review");
      } else {
        if(check)await page.getByRole("checkbox",{name:check}).check();
        await page.getByRole("button",{name:"Load example"}).click();
        await page.waitForFunction(()=>[...document.querySelectorAll("textarea")].some(e=>e.value.length>0));
        await page.getByRole("button",{name:"Review draft"}).click();card=page.locator(".writing-review");
      }
      await card.waitFor({state:"visible"});
      const reports=[await card.evaluate(audit)];
      const disclosures=card.locator("details>summary");
      for(let i=0;i<await disclosures.count();i++) {
        const control=disclosures.nth(i);
        if(!await control.isVisible())continue;
        await control.click();reports.push(await card.evaluate(audit));await control.click();
      }
      const evidence={};
      for(const r of reports)for(const [key,value] of Object.entries(r)) {
        const total=evidence[key]??{nodes:0,min:Infinity};
        total.nodes+=value.nodes;total.min=Math.min(total.min,value.min);evidence[key]=total;
      }
      if(mode==="summary"&&!evidence.link)throw new Error("Missing visible study source link");
      observed.push({mode,theme,width,readable:evidence});
    }
  }
  return observed;
}
