      const el = (id) => document.getElementById(id);
      const paperTitleEl = el("paper-title");
      const sessionTitleEl = el("session-title");
      const speakersEl = el("speakers");
      const dateEl = el("date");
      const timeEl = el("time");
      const durationEl = el("duration");
      const tzEl = el("tz");
      const tzCustomEl = el("tz-custom");
      const paperUrlEl = el("paper-url");
      const meetUrlEl = el("meet-url");
      const meetingNoteEl = el("meeting-note");
      const descriptionEl = el("description");
      const notesEl = el("notes");
      const displayDateEl = el("display-date");
      const displayTimeEl = el("display-time");
      const titlePrefixEl = el("title-prefix");
      const maxDetailsEl = el("max-details");
      const gcalUrlEl = el("gcal-url");
      const gcalMetaEl = el("gcal-meta");
      const openGcalEl = el("open-gcal");
      const shortenerEl = el("shortener");
      const shortUrlEl = el("short-url");
      const shortStatusEl = el("short-status");
      const openShortenerEl = el("open-shortener");
      const prTextEl = el("pr-text");
      const repoEl = el("repo");
      const baseBranchEl = el("base-branch");
      const yamlPathEl = el("yaml-path");
      const prTitleEl = el("pr-title");
      const prBodyEl = el("pr-body");
      const prStatusEl = el("pr-status");
      const prUrlEl = el("pr-url");
      const yamlOutputEl = el("yaml-output");
      const prCommandEl = el("pr-command");
      const prPayloadEl = el("pr-payload");
      const imageUrlEl = el("image-url");
      const imageAltEl = el("image-alt");
      const imageFileEl = el("image-file");
      const imagePreviewEl = el("image-preview");

      const DEFAULT_EVENT_PREFIX = "LeMaterial Reading Group - ";

      const safeTrim = (value) => (value || "").trim();

      function getTimeZone() {
        if (tzEl.value === "custom") {
          return safeTrim(tzCustomEl.value) || "UTC";
        }
        return tzEl.value;
      }

      function formatUtcBasic(date) {
        const pad = (num) => String(num).padStart(2, "0");
        return (
          date.getUTCFullYear() +
          pad(date.getUTCMonth() + 1) +
          pad(date.getUTCDate()) +
          "T" +
          pad(date.getUTCHours()) +
          pad(date.getUTCMinutes()) +
          pad(date.getUTCSeconds()) +
          "Z"
        );
      }

      function getTimeZoneOffsetMinutes(date, timeZone) {
        const dtf = new Intl.DateTimeFormat("en-US", {
          timeZone,
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        });
        const parts = dtf.formatToParts(date);
        const values = {};
        for (const part of parts) {
          if (part.type !== "literal") {
            values[part.type] = part.value;
          }
        }
        const asUTC = Date.UTC(
          Number(values.year),
          Number(values.month) - 1,
          Number(values.day),
          Number(values.hour),
          Number(values.minute),
          Number(values.second)
        );
        return (asUTC - date.getTime()) / 60000;
      }

      function computeUtcRange(dateStr, timeStr, durationMinutes, timeZone) {
        const [year, month, day] = dateStr.split("-").map(Number);
        const [hour, minute] = timeStr.split(":").map(Number);
        const utcCandidate = Date.UTC(year, month - 1, day, hour, minute, 0);
        const offsetMinutes = getTimeZoneOffsetMinutes(new Date(utcCandidate), timeZone);
        const startUtcMillis = utcCandidate - offsetMinutes * 60000;
        const endUtcMillis = startUtcMillis + durationMinutes * 60000;
        return {
          start: new Date(startUtcMillis),
          end: new Date(endUtcMillis),
        };
      }

      function trimDescription(description, staticText, maxLen) {
        if (!description) {
          return { text: "", trimmed: false };
        }
        const separator = staticText ? "\n\n" : "";
        const available = maxLen - staticText.length - separator.length;
        if (available <= 0) {
          return { text: "", trimmed: true };
        }
        if (description.length <= available) {
          return { text: description, trimmed: false };
        }
        let sliced = description.slice(0, available);
        sliced = sliced.replace(/\s+\S*$/, "").trimEnd();
        if (!sliced) {
          sliced = description.slice(0, Math.max(0, available - 3)).trimEnd();
        }
        return { text: `${sliced}...`, trimmed: true };
      }

      function buildDetails({ description, speakers, paperTitle, paperUrl, meetUrl, notes, maxLen }) {
        const parts = [];
        if (description) {
          parts.push(description);
        }
        if (speakers) {
          parts.push(`Speakers: ${speakers}`);
        }
        if (paperTitle && paperUrl) {
          parts.push(`Paper: ${paperTitle} (${paperUrl})`);
        } else if (paperTitle) {
          parts.push(`Paper: ${paperTitle}`);
        } else if (paperUrl) {
          parts.push(`Paper: ${paperUrl}`);
        }
        if (meetUrl) {
          parts.push(`Meeting: ${meetUrl}`);
        }
        if (notes) {
          parts.push(notes);
        }

        let details = parts.join("\n\n");
        let trimmed = false;
        if (maxLen && details.length > maxLen && description) {
          const staticParts = parts.slice(1);
          const staticText = staticParts.join("\n\n");
          const trimmedDesc = trimDescription(description, staticText, maxLen);
          trimmed = trimmedDesc.trimmed;
          const rebuilt = [];
          if (trimmedDesc.text) {
            rebuilt.push(trimmedDesc.text);
          }
          if (staticText) {
            rebuilt.push(staticText);
          }
          details = rebuilt.join("\n\n");
        }
        if (maxLen && details.length > maxLen) {
          details = details.slice(0, maxLen);
          trimmed = true;
        }
        return { details, trimmed };
      }

      function buildGcalUrl(data) {
        if (!data.title || !data.date || !data.time) {
          return { url: "", error: "Add a title, date, and start time to build the link." };
        }
        let range;
        try {
          range = computeUtcRange(data.date, data.time, data.duration, data.timeZone);
        } catch (err) {
          return { url: "", error: "Invalid time zone. Check the custom time zone value." };
        }
        const { details, trimmed } = buildDetails({
          description: data.description,
          speakers: data.speakers,
          paperTitle: data.paperTitle,
          paperUrl: data.paperUrl,
          meetUrl: data.meetUrl,
          notes: data.notes,
          maxLen: data.maxDetails,
        });

        const params = new URLSearchParams({
          action: "TEMPLATE",
          text: `${data.titlePrefix}${data.title}`,
          dates: `${formatUtcBasic(range.start)}/${formatUtcBasic(range.end)}`,
          stz: data.timeZone,
          etz: data.timeZone,
        });
        if (details) {
          params.set("details", details);
        }
        if (data.meetUrl) {
          params.set("location", data.meetUrl);
        }

        return {
          url: `https://calendar.google.com/calendar/r/eventedit?${params.toString()}`,
          trimmed,
        };
      }

      function buildPrText(data, gcalUrl, shortUrl) {
        const lines = [];
        lines.push("LeMaterial Reading Group - Next Session");
        lines.push("");
        if (data.sessionTitle) lines.push(data.sessionTitle);
        if (data.paperTitle && data.paperTitle !== data.sessionTitle) {
          lines.push(`Paper title: ${data.paperTitle}`);
        }
        if (data.speakers) lines.push(`Speaker(s): ${data.speakers}`);

        const whenParts = [];
        if (data.displayDate) whenParts.push(data.displayDate);
        if (data.displayTime) whenParts.push(data.displayTime);
        if (whenParts.length) {
          lines.push(`When: ${whenParts.join(" - ")}`);
        }

        if (data.paperUrl) lines.push(`Paper: ${data.paperUrl}`);
        if (data.meetUrl) lines.push(`Join: ${data.meetUrl}`);
        if (shortUrl) {
          lines.push(`Add to calendar: ${shortUrl}`);
        } else if (gcalUrl) {
          lines.push(`Add to calendar: ${gcalUrl}`);
        }
        if (data.imageUrl) lines.push(`Image: ${data.imageUrl}`);
        if (data.description) {
          lines.push("");
          lines.push(data.description);
        }
        return lines.join("\n");
      }

      function updatePreviewFromFile(file) {
        if (!file) {
          imagePreviewEl.style.display = "none";
          imagePreviewEl.src = "";
          return;
        }
        const reader = new FileReader();
        reader.onload = (event) => {
          imagePreviewEl.src = event.target.result;
          imagePreviewEl.style.display = "block";
        };
        reader.readAsDataURL(file);
      }

      function getDisplayDate(dateStr) {
        if (!dateStr) return "";
        const date = new Date(`${dateStr}T00:00:00`);
        if (Number.isNaN(date.getTime())) return "";
        return date.toLocaleDateString("en-US", {
          weekday: "long",
          year: "numeric",
          month: "long",
          day: "numeric",
        });
      }

      function getNextWeekday(targetDay) {
        const date = new Date();
        date.setHours(0, 0, 0, 0);
        const currentDay = date.getDay();
        let delta = (targetDay - currentDay + 7) % 7;
        if (delta === 0) delta = 7;
        date.setDate(date.getDate() + delta);
        return date;
      }

      function setDefaultDateIfEmpty() {
        if (dateEl.value) return;
        const nextWednesday = getNextWeekday(3);
        dateEl.value = nextWednesday.toISOString().slice(0, 10);
      }

      function updateOutputs() {
        const paperTitle = safeTrim(paperTitleEl.value);
        const sessionTitle = safeTrim(sessionTitleEl.value) || paperTitle;
        const data = {
          title: sessionTitle,
          sessionTitle,
          paperTitle,
          speakers: safeTrim(speakersEl.value),
          date: dateEl.value,
          time: timeEl.value,
          duration: Math.max(15, Number(durationEl.value || 60)),
          timeZone: getTimeZone(),
          paperUrl: safeTrim(paperUrlEl.value),
          meetUrl: safeTrim(meetUrlEl.value),
          meetingNote: safeTrim(meetingNoteEl.value),
          description: safeTrim(descriptionEl.value),
          notes: safeTrim(notesEl.value),
          displayDate: safeTrim(displayDateEl.value) || getDisplayDate(dateEl.value),
          displayTime: safeTrim(displayTimeEl.value),
          titlePrefix: titlePrefixEl.value || DEFAULT_EVENT_PREFIX,
          maxDetails: Math.max(200, Number(maxDetailsEl.value || 900)),
          imageUrl: safeTrim(imageUrlEl.value),
          imageAlt: safeTrim(imageAltEl.value),
        };

        const gcal = buildGcalUrl(data);
        gcalUrlEl.value = gcal.url || "";
        openGcalEl.href = gcal.url || "#";
        openGcalEl.style.pointerEvents = gcal.url ? "auto" : "none";
        openGcalEl.style.opacity = gcal.url ? "1" : "0.5";

        if (gcal.error) {
          gcalMetaEl.textContent = gcal.error;
          gcalMetaEl.className = "status danger";
        } else if (gcal.url) {
          const length = gcal.url.length;
          const lengthMsg = `URL length: ${length}`;
          if (length > 1900) {
            gcalMetaEl.textContent = `${lengthMsg} (long - consider trimming description or using a shortener).`;
            gcalMetaEl.className = "status warn";
          } else {
            gcalMetaEl.textContent = `${lengthMsg}${gcal.trimmed ? " (description trimmed)" : ""}.`;
            gcalMetaEl.className = "status";
          }
        } else {
          gcalMetaEl.textContent = "";
        }

        const shortValue = shortUrlEl.value.trim();
        prTextEl.value = buildPrText(data, gcal.url, shortValue);

        if (!prTitleEl.value.trim()) {
          prTitleEl.value = sessionTitle ? `Reading group session: ${sessionTitle}` : "Reading group session update";
        }
        if (!prBodyEl.value.trim()) {
          prBodyEl.value = prTextEl.value;
        }

        const payload = {
          repo: safeTrim(repoEl.value),
          baseBranch: safeTrim(baseBranchEl.value) || "main",
          path: safeTrim(yamlPathEl.value) || "data/reading-group/next_session.yaml",
          prTitle: safeTrim(prTitleEl.value),
          prBody: safeTrim(prBodyEl.value),
          gcalUrl: gcal.url,
          shortUrl: shortValue,
          calendarLink: shortValue || gcal.url,
          session: {
            paperTitle,
            sessionTitle,
            speakers: safeTrim(speakersEl.value),
            date: dateEl.value,
            displayDate: safeTrim(displayDateEl.value) || getDisplayDate(dateEl.value),
            displayTime: safeTrim(displayTimeEl.value),
            timeZone: getTimeZone(),
            time: timeEl.value,
            durationMinutes: Math.max(15, Number(durationEl.value || 60)),
            paperUrl: safeTrim(paperUrlEl.value),
            meetUrl: safeTrim(meetUrlEl.value),
            meetingNote: safeTrim(meetingNoteEl.value),
            description: safeTrim(descriptionEl.value),
            notes: safeTrim(notesEl.value),
            imageUrl: safeTrim(imageUrlEl.value),
            imageAlt: safeTrim(imageAltEl.value),
          },
        };
        prPayloadEl.value = JSON.stringify(payload, null, 2);
        prCommandEl.value = "pbpaste | node scripts/reading-group-pr.js --payload -";
      }

      async function copyValue(value) {
        if (!value) return;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(value);
          return;
        }
        const helper = document.createElement("textarea");
        helper.value = value;
        document.body.appendChild(helper);
        helper.select();
        document.execCommand("copy");
        document.body.removeChild(helper);
      }

      async function pasteShortUrl() {
        if (!navigator.clipboard || !navigator.clipboard.readText) {
          shortStatusEl.textContent = "Clipboard API unavailable. Paste manually.";
          shortStatusEl.className = "status warn";
          return;
        }
        try {
          const text = (await navigator.clipboard.readText()).trim();
          if (!text) {
            shortStatusEl.textContent = "Clipboard is empty.";
            shortStatusEl.className = "status warn";
            return;
          }
          shortUrlEl.value = text;
          shortStatusEl.textContent = "Short URL pasted.";
          shortStatusEl.className = "status";
          updateOutputs();
        } catch (err) {
          shortStatusEl.textContent = "Clipboard read failed. Paste manually.";
          shortStatusEl.className = "status warn";
        }
      }

      function getFunctionBase() {
        if (window.location.hostname === "localhost" && window.location.port === "1313") {
          return "http://localhost:8888";
        }
        return "";
      }

      async function shortenViaBackend(longUrl) {
        try {
          const res = await fetch(`${getFunctionBase()}/.netlify/functions/reading-group-pr`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action: "shorten",
              url: longUrl,
              shortener: shortenerEl.value,
            }),
          });
          if (!res.ok) return null;
          const data = await res.json();
          if (data && data.shortUrl) return data.shortUrl;
        } catch (err) {
          return null;
        }
        return null;
      }

      async function shortenUrl() {
        shortStatusEl.textContent = "";
        const longUrl = gcalUrlEl.value.trim();
        if (!longUrl) {
          shortStatusEl.textContent = "Generate the calendar URL first.";
          shortStatusEl.className = "status danger";
          return;
        }
        const service = shortenerEl.value;
        if (service === "manual") {
          shortStatusEl.textContent = "Manual mode selected. Open the shortener to generate a link.";
          shortStatusEl.className = "status";
          return;
        }

        const backendShort = await shortenViaBackend(longUrl);
        if (backendShort) {
          shortUrlEl.value = backendShort;
          shortStatusEl.textContent = "Short URL generated via backend.";
          shortStatusEl.className = "status";
          updateOutputs();
          return;
        }

        let endpoint = "";
        if (service === "isgd") {
          endpoint = `https://is.gd/create.php?format=simple&url=${encodeURIComponent(longUrl)}`;
        } else if (service === "tinyurl") {
          endpoint = `https://tinyurl.com/api-create.php?url=${encodeURIComponent(longUrl)}`;
        }

        if (service === "isgd") {
          openShortenerEl.href = "https://is.gd/";
        } else if (service === "tinyurl") {
          openShortenerEl.href = "https://tinyurl.com/";
        } else {
          openShortenerEl.href = "#";
        }

        try {
          const res = await fetch(endpoint, { mode: "cors" });
          if (!res.ok) throw new Error("Request failed");
          const text = (await res.text()).trim();
          if (!text || text.startsWith("Error")) throw new Error(text || "Shorten failed");
          shortUrlEl.value = text;
          shortStatusEl.textContent = "Short URL generated.";
          shortStatusEl.className = "status";
        } catch (err) {
          shortStatusEl.textContent = "Shorten failed in browser. Use 'Open shortener' and copy the result.";
          shortStatusEl.className = "status warn";
        }

        updateOutputs();
      }

      async function createPr() {
        prStatusEl.textContent = "Creating PR...";
        prStatusEl.className = "status";
        prUrlEl.value = "";
        yamlOutputEl.value = "";

        const payload = {
          action: "create_pr",
          repo: safeTrim(repoEl.value),
          baseBranch: safeTrim(baseBranchEl.value) || "main",
          path: safeTrim(yamlPathEl.value) || "data/reading-group/next_session.yaml",
          prTitle: safeTrim(prTitleEl.value),
          prBody: safeTrim(prBodyEl.value),
          gcalUrl: gcalUrlEl.value.trim(),
          shortener: shortenerEl.value,
          shortUrl: shortUrlEl.value.trim(),
          session: {
            paperTitle: safeTrim(paperTitleEl.value),
            sessionTitle: safeTrim(sessionTitleEl.value),
            speakers: safeTrim(speakersEl.value),
            date: dateEl.value,
            displayDate: safeTrim(displayDateEl.value) || getDisplayDate(dateEl.value),
            displayTime: safeTrim(displayTimeEl.value),
            timeZone: getTimeZone(),
            time: timeEl.value,
            durationMinutes: Math.max(15, Number(durationEl.value || 60)),
            paperUrl: safeTrim(paperUrlEl.value),
            meetUrl: safeTrim(meetUrlEl.value),
            meetingNote: safeTrim(meetingNoteEl.value),
            description: safeTrim(descriptionEl.value),
            notes: safeTrim(notesEl.value),
            imageUrl: safeTrim(imageUrlEl.value),
            imageAlt: safeTrim(imageAltEl.value),
          },
        };

        try {
          const res = await fetch(`${getFunctionBase()}/.netlify/functions/reading-group-pr`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await res.json();
          if (!res.ok || !data || !data.ok) {
            const message = (data && data.error) || "PR creation failed.";
            prStatusEl.textContent = message;
            prStatusEl.className = "status danger";
            return;
          }
          if (data.shortUrl) {
            shortUrlEl.value = data.shortUrl;
          }
          if (data.calendarLink) {
            gcalUrlEl.value = data.calendarLink;
          }
          prUrlEl.value = data.prUrl || "";
          yamlOutputEl.value = data.updatedYaml || "";
          prStatusEl.textContent = data.prUrl ? `PR created: ${data.prUrl}` : "PR created.";
          prStatusEl.className = "status";
          updateOutputs();
        } catch (err) {
          prStatusEl.textContent = "PR creation failed. Check your Netlify function logs.";
          prStatusEl.className = "status danger";
        }
      }

      async function copyPayload() {
        await copyValue(prPayloadEl.value);
        prStatusEl.textContent = "PR payload copied.";
        prStatusEl.className = "status";
      }

      tzEl.addEventListener("change", () => {
        tzCustomEl.style.display = tzEl.value === "custom" ? "block" : "none";
        updateOutputs();
      });

      [
        paperTitleEl,
        sessionTitleEl,
        speakersEl,
        dateEl,
        timeEl,
        durationEl,
        tzCustomEl,
        paperUrlEl,
        meetUrlEl,
        meetingNoteEl,
        descriptionEl,
        notesEl,
        displayDateEl,
        displayTimeEl,
        titlePrefixEl,
        maxDetailsEl,
        imageUrlEl,
        imageAltEl,
        shortUrlEl,
      ].forEach((input) => input.addEventListener("input", updateOutputs));

      el("copy-gcal").addEventListener("click", () => copyValue(gcalUrlEl.value));
      el("copy-short").addEventListener("click", () => copyValue(shortUrlEl.value));
      el("paste-short").addEventListener("click", pasteShortUrl);
      el("copy-pr").addEventListener("click", () => copyValue(prTextEl.value));
      el("shorten").addEventListener("click", shortenUrl);
      el("create-pr").addEventListener("click", createPr);
      el("copy-payload").addEventListener("click", copyPayload);

      imageFileEl.addEventListener("change", (event) => {
        const file = event.target.files && event.target.files[0];
        updatePreviewFromFile(file);
      });

      setDefaultDateIfEmpty();
      updateOutputs();
