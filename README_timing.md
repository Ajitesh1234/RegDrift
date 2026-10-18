## Is the circular still in force?

A finding is only worth a ticket if the circular applies today, so RegDrift reads the dates before it reads the rules
(no LLM involved, so it is tested exactly):

| Status | How it is detected | What happens |
|---|---|---|
| **Withdrawn** | the diagonal "Withdrawn" stamp RBI prints on withdrawn PDFs (read from the slanted text on the first 3 pages) | nothing is checked, no tickets; `--allow-withdrawn` overrides |
| **Expired** | a temporary measure whose "until / till <date>" is in the past | findings are shown, no tickets are filed |
| **Upcoming** | the general commencement sentence ("These Directions shall come into effect from 1 April 2027") is in the future | findings are kept; the ticket says when it takes effect |
| **In force** | everything else | normal |

It also warns when a circular is 3 or more years old (measured from the latest "Updated as on" date for Master
Directions), when some paragraphs start later than the rest, and when no issue date can be found.

Body sentences such as "circular X stands withdrawn" are deliberately ignored, because live circulars say that about
*other* documents all the time.

**What it cannot know:** whether a *newer* circular amended or replaced this one. It only sees the file it is given. A PDF
downloaded before it was withdrawn carries no stamp, and a 2009 circular may look fine while a 2023 Master Direction has
replaced it. The age warning is there for that case: confirm on rbi.org.in. Scanned stamps (images) are not read, and dates
are read day-first (03/04/2027 is 3 April).
