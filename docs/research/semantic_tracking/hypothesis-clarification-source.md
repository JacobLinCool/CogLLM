# Public hypothesis questioning, response, and original questioner confirmation

The public data of Hypothesis can link a question posted in the main text, another user's response, and the original asker's confirmation of that response. Two structurally complete discussions were found. one of them links to a textbook chapter with an explicitly stated open license. The verifiable time is determined by the timestamps of three posts, allowing observation of the order of the post and response. To determine which parts of the main text readers had seen at the time of posting, an additional exposure record is required. [Hypothesis public API](https://web.hypothes.is/developers/) · [Public CC0 attribution label](https://web.hypothes.is/help/what-is-the-license-on-annotations/)

## Two denominators

Search `GET /api/search` with `group=__world__` fixed and use an acknowledgment phrase to identify possible examples. `any=thanks for clarifying` returned 18 records, including 12 replies carrying `references` whose body contained the phrase. Two records included an original question anchored to the text, a direct reply from another account, and a later acknowledgment from the original account. Both matched on URI and appeared in the expected chronological order. Because the search starts from an explicit acknowledgment, it cannot estimate the proportion of questions that are answered overall. [API search method](https://web.hypothes.is/help/how-hypothesis-search-works/)

Retrieve all publicly available records for each page using the precise URI. this step does not apply the confirmation phrase filter. The original question candidates are selected based on the interrogative wording of the root comment and `TextQuoteSelector`, `TextPositionSelector`, and counted for other accounts' replies, the original account replying to another person, and root comments containing the fixed confirmation phrase in the latter. Two primary pages are identified from the above cases, along with adjacent sections of the textbook. [Count and Conditions](../../../research/results/hypothesis-clarification-source-audit.json)

| Article URI | Publicly disclosed | Root comment | There are textual anchors and interrogative phrasing. | There is another response. | The original inquirer followed up. | There is a fixed confirmation phrase. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open RN 3.5 | 24 | 22 | 20 | 1 | 1 | 1 |
| Open RN 3.4 | 0 | 0 | 0 | 0 | 0 | 0 |
| Open RN 3.6 | 2 | 2 | 2 | 0 | 0 | 0 |
| Marginal Syllabus PDF | 273 | 186 | 36 | 12 | 3 | 1 |

## Scope verification

- **Open RN 3.5.** The original chapter explicitly states CC BY 4.0, with noted exceptions. The five-character selection length and position selector are consistent. This text appears only once in the blank normalization text of the same-version NCBI chapter, but the surrounding strings of the selector do not fully match. Requests for the original HTML and official XHTML, XML exports all return HTTP 403. the original DOM character coordinates still require character-by-character verification. [Publisher Chapter and License](https://wtcs.pressbooks.pub/medterm/chapter/3-5-diseases-disorders-and-injuries-of-the-integumentary-system/) · [NCBI Same-Version Chapter](https://www.ncbi.nlm.nih.gov/sites/books/NBK607449/)
- **Marginal Syllabus.** The original problem's text selector and position width are both 15 characters. The public license for three messages has been confirmed. The independent reuse terms and original offset for the PDF remain unverified, retained as evidence of interactive data. [Project and Data Type](https://education.ucdenver.edu/partnerships/our-impact/marginal-syllabus) · [Original PDF](https://educatorinnovator.org/wp-content/uploads/2018/11/What_s_Radical_marginal-syllabus.pdf)

If more cases with authorization and verifiable original text are found, one can test whether the model ranks the original asker's confirmed response before responses to other questions within the same text when given the same question and anchor span. `references`, `created`, and account similarity establish an observed three-turn relationship. Readers who did not comment or reply again have an unobserved understanding state. This audit currently confirms only two types of public interaction connections. corresponding observations for reading stop, group resolution rate, and per-token resolution are still missing.

[Original research record](../../../research/semantic_tracking/hypothesis-clarification-source.md)
