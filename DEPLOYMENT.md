# Putting Ground Craft AI Online — A Guide for Non-Coders

You do **not** need to know how to code to follow this. You will never open a code editor, never use a "terminal", and never type a command. Everything happens in your web browser, by clicking buttons and copying and pasting text.

You will edit exactly **one line** of one file, and I will show you precisely which line.

---

## What you're about to do

Your app has two halves that live in two different places:

- **The front** — the pages people see and click. Goes on a service called **Vercel**.
- **The engine** — the part that talks to the AI. Goes on a service called **Render**.

Then you tell the two halves each other's addresses so they can talk. That's it.

**Time:** about 60–90 minutes the first time, most of it waiting.
**Cost:** nothing. Every service below has a free level that's enough for this app.

> **A note on free plans:** these companies change their free offerings from time to time. If a screen looks different from what's described here, or asks for a card, check the company's current pricing page before continuing.

---

## Five words you'll see, in plain English

| Word | What it actually means |
|---|---|
| **Repository** (or "repo") | A folder of your files, stored on GitHub. Think of it as a shared Dropbox folder that Vercel and Render can both read from. |
| **Deploy** | Take the files and make them live on the internet at a real web address. |
| **Environment variable** | A secret setting you type into a website's dashboard instead of writing it in a file. This is where your AI key goes, so it never ends up in your public files. |
| **API key** | A long password that lets your app use an AI service. Treat it like a bank PIN. |
| **Backend / Frontend** | Backend = the engine. Frontend = the pages people see. |

---

# PART 1 · Create your accounts

You need three. All free, all sign-up-with-email.

### 1.1 GitHub — where your files live

1. Go to **github.com**
2. Click **Sign up**
3. Enter your email, pick a password and a username, verify your email
4. When it asks about plans, choose the **Free** one

### 1.2 OpenRouter — where the AI comes from

1. Go to **openrouter.ai**
2. Click **Sign in** (top right) — you can use your Google account
3. That's all for now. You'll get the key in Part 2.

### 1.3 Render — where the engine lives

1. Go to **render.com**
2. Click **Get Started**
3. Choose **Sign up with GitHub** — this links the two accounts, which saves work later
4. Approve the permission screen GitHub shows you

You'll create the Vercel account later, in Part 5.

---

# PART 2 · Get your AI key

This is the key that lets your app actually talk to an AI. **This is the step people most often get wrong, so go slowly.**

1. Go to **openrouter.ai** and sign in
2. Click your **profile picture** (top right) → **Keys**
3. Click **Create Key**
4. In the name box type: `groundcraft`
5. Leave the credit limit box empty
6. Click **Create**
7. A long line of text appears, starting with `sk-or-v1-...`

**Copy it now and paste it somewhere safe** — a note on your phone, an email draft to yourself, anywhere you can find it again in ten minutes.

> ⚠️ **You only get to see this once.** When you close that box, OpenRouter will never show you the key again. If you lose it, no harm done — just delete that key and create another.

> 🔒 **Never** put this key in a message, a screenshot, or a public file. Anyone who has it can spend your AI budget. It only ever goes in one place: the Render settings screen in Part 4.

### What "free AI" means here

Your app is set up to use AI models that cost nothing. There are limits — roughly 50 questions a day across everyone using your app, and the app itself limits each visitor to 10 questions an hour so one person can't use up the day's supply.

If your app gets popular and you want to lift that, adding $10 of credit to your OpenRouter account raises the daily limit substantially. You do not need to do this to launch.

---

# PART 3 · Put your files on GitHub

### 3.1 Unzip the project

1. Find the file `groundcraft-ai.zip` you downloaded
2. Double-click it to unzip
3. You should now have a folder containing: `backend`, `frontend`, `README.md`, and a few others

### 3.2 Create the repository

1. Go to **github.com** and sign in
2. Click the **+** in the top right → **New repository**
3. **Repository name:** `groundcraft-ai`
4. Choose **Public**
5. Do **not** tick "Add a README file"
6. Click **Create repository**

### 3.3 Upload the files

1. On the page that appears, click the link **uploading an existing file**
2. Open the unzipped folder on your computer
3. Select **everything inside it** (Ctrl+A on Windows, Cmd+A on Mac) and **drag it all onto the browser window**
4. Wait for the upload — the file list should show `backend`, `frontend` and the rest
5. Scroll to the bottom, click the green **Commit changes** button

### ✅ Check it worked

Your page should now list a `backend` folder and a `frontend` folder. Click into `backend` — you should see a file called `requirements.txt`. If you see that, you're good.

> **If you only see loose files and no folders:** you dragged the *contents* of the wrong level, or the outer folder itself. Delete the repository (Settings → scroll to bottom → Delete) and redo step 3.3, making sure you're selecting the items *inside* the unzipped folder.

---

# PART 4 · Put the engine online (Render)

### 4.1 Create the service

1. Go to **render.com** and sign in
2. Click **New +** (top right) → **Web Service**
3. Find `groundcraft-ai` in the repository list → click **Connect**
   - If you don't see it, click **Configure account** and give Render permission to see your repositories

### 4.2 Fill in the settings

Type these **exactly**. The Root Directory one is the most commonly missed.

| Field | What to enter |
|---|---|
| **Name** | `groundcraft-api` |
| **Region** | Whichever is nearest you |
| **Branch** | `main` |
| **Root Directory** | `backend` ← **don't skip this** |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | **Free** |

### 4.3 Add your AI key

Still on the same page, scroll down to **Environment Variables** and click **Add Environment Variable**:

- **Key:** `OPENROUTER_API_KEY`
- **Value:** paste the `sk-or-v1-...` key you saved in Part 2

Double-check there's no extra space before or after the pasted key. A stray space is the single most common cause of "it doesn't work".

### 4.4 Deploy

1. Click **Create Web Service**
2. Wait 3–5 minutes. Text will scroll past — that's normal, ignore it.
3. When you see **Live** in green at the top, it's done
4. **Copy the web address** at the top of the page. It looks like `https://groundcraft-api.onrender.com`. **Save this — you need it in Part 6.**

### ✅ Check it worked

Take your address, add `/api/health` to the end, and open it in a new tab:

```
https://groundcraft-api.onrender.com/api/health
```

You should see: `{"status":"ok"}`

That's your engine running. If you see an error instead, jump to Troubleshooting at the bottom.

---

# PART 5 · Put the pages online (Vercel)

1. Go to **vercel.com**
2. Click **Sign Up** → **Continue with GitHub** → approve
3. On your dashboard click **Add New...** → **Project**
4. Find `groundcraft-ai` → click **Import**
5. Set these:
   - **Framework Preset:** `Other`
   - **Root Directory:** click **Edit**, choose the **`frontend`** folder, click **Continue**
   - Leave Build and Output settings empty
6. Click **Deploy**
7. Wait about a minute, then click **Continue to Dashboard**
8. **Copy your app's address** at the top — something like `https://groundcraft-ai.vercel.app`. **Save this too.**

### ✅ Check it worked

Open your Vercel address. The app should appear, looking correct, with the logo and mission tiles.

**It won't answer questions yet** — the two halves haven't been introduced. That's Part 6.

---

# PART 6 · Introduce the two halves

This is the one and only file you'll edit. It has a single line in it.

### 6.1 Tell the pages where the engine is

1. Go to your GitHub repository
2. Click the **`frontend`** folder
3. Click the file **`config.js`**
4. Click the **pencil icon** (✏️) near the top right
5. Find the last line. It says:

   ```
   window.GROUNDCRAFT_API_BASE = "http://localhost:8000";
   ```

6. Replace `http://localhost:8000` with your Render address from Part 4, keeping the quote marks:

   ```
   window.GROUNDCRAFT_API_BASE = "https://groundcraft-api.onrender.com";
   ```

   **Three things to get right:**
   - Keep both `"` quote marks
   - Keep the `;` at the end
   - **No slash at the end** of the address

7. Click the green **Commit changes...** button → **Commit changes**

Vercel notices the change and updates your app automatically within about a minute.

### 6.2 Tell the engine which pages to trust

1. Go to **render.com** → click your `groundcraft-api` service
2. Click **Environment** in the left menu
3. Click **Add Environment Variable**:
   - **Key:** `ALLOWED_ORIGINS`
   - **Value:** your Vercel address, e.g. `https://groundcraft-ai.vercel.app` (again, no slash at the end)
4. Click **Save Changes** — Render restarts itself, taking a minute or two

This stops other websites from using your AI key. Don't skip it.

### ✅ The real test

1. Open your Vercel address
2. Tap **Start learning** → tap the first mission → **Start mission**
3. Press **Ask**
4. **Wait up to a minute.** The first question after a quiet spell is slow — see the note below.
5. Two answers should appear side by side

**🎉 If you see two answers, you are live and finished.** Everything below is optional.

> ### Why the first question is slow
> On the free plan, Render puts your engine to sleep when nobody's used it for about 15 minutes. The next person to arrive has to wait 30–60 seconds while it wakes up. After that it's fast again.
>
> This is normal and not a fault. It's worth telling visitors — otherwise they'll assume the app is broken and leave.

---

# PART 7 · Optional: let people save their progress

Skip this and everything still works — people just start fresh each visit and the leaderboard empties whenever the engine restarts.

Doing this takes about 20 minutes and involves two more free accounts.

### 7.1 Google Sign-In

1. Go to **console.cloud.google.com** and sign in
2. Top left, click the project dropdown → **New Project** → name it `Ground Craft AI` → **Create**
3. In the search bar type `OAuth consent screen`, open it
4. Choose **External** → **Create**
5. Fill in App name (`Ground Craft AI`), your email for both support and developer contact → **Save and Continue** through the remaining screens
6. In the search bar type `Credentials`, open it
7. **Create Credentials** → **OAuth client ID**
8. **Application type:** `Web application`
9. Under **Authorised JavaScript origins** click **Add URI** and paste your Vercel address (no slash at the end)
10. Click **Create**
11. Copy the **Client ID** — a long string ending in `.apps.googleusercontent.com`
12. Go to Render → your service → **Environment** → add:
    - **Key:** `GOOGLE_CLIENT_ID`
    - **Value:** the client ID you copied
13. **Save Changes**

### 7.2 The database

1. Go to **supabase.com** → **Start your project** → sign in with GitHub
2. **New Project** → name it `groundcraft` → choose a region → set a database password (save it somewhere) → **Create**
3. Wait ~2 minutes for it to finish setting up
4. In the left menu click **SQL Editor** → **New query**
5. Copy this **entire block** and paste it in:

```sql
create table feedback (
  id bigint generated always as identity primary key,
  name text not null default 'Anonymous',
  emoji text not null default '🙂',
  text text not null,
  created_at timestamptz not null default now()
);

create table visits (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now()
);

create table progress (
  user_sub text primary key,
  name text not null,
  email text not null,
  xp int not null default 0,
  completed jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now()
);
```

6. Click **Run**. You should see "Success".
7. Left menu → **Project Settings** (gear icon) → **API**
8. Copy two things:
   - **Project URL**
   - Under Project API keys, the one labelled **`anon` `public`** ← **this one only**

> ⚠️ There's a second key on that page called **service_role**. **Never use it.** It bypasses all security. You want the one marked `anon public`.

9. Go to Render → **Environment** → add both:
   - **Key:** `SUPABASE_URL` — **Value:** the Project URL
   - **Key:** `SUPABASE_KEY` — **Value:** the `anon public` key
10. **Save Changes**

### ✅ Check it worked

Open `https://your-render-address.onrender.com/api/stats` in a browser. It should say `"storage":"supabase"`. If it says `"in-memory"`, one of the two values is wrong or has a stray space.

Then: open your app, sign in with Google, finish a mission, close the tab entirely, reopen — your XP should still be there.

---

# Before you share the link

- [ ] The app opens at your Vercel address
- [ ] Pressing **Ask** gives two answers
- [ ] You've opened it on a **phone** and tapped through a few missions
- [ ] You've tried the **dark/light** toggle (top right)
- [ ] You've tapped the **lightbulb** button and asked the assistant a question
- [ ] `ALLOWED_ORIGINS` is set in Render (Part 6.2)
- [ ] You've mentioned the slow first load somewhere people will see it

---

# Troubleshooting

### "Cannot reach the Ground Craft AI backend"

The two halves aren't talking. In order of likelihood:

1. **A slash at the end** of the address in `config.js`. Remove it.
2. **`http` instead of `https`** in `config.js`. Render addresses use `https`.
3. **A typo** in the address. Compare it letter by letter with the one on your Render dashboard.
4. **`ALLOWED_ORIGINS` doesn't match** your Vercel address exactly. Compare those too.
5. **Vercel hasn't updated yet.** Check the Deployments tab on Vercel — the newest one should say Ready.

### The first question takes forever, then works

Normal. Free-plan sleep, explained in Part 6. Nothing to fix.

### Every question fails with an error mentioning 401 or "unauthorized"

Your AI key is wrong. Go to Render → Environment → `OPENROUTER_API_KEY`, delete the value, and paste it again carefully. If you lost the key, make a new one (Part 2) — old keys can't be recovered.

### Errors mentioning 429 or "rate limit"

You've used the day's free AI allowance. Wait, or add $10 credit at OpenRouter to raise the ceiling.

### The Sign in button says it isn't configured

`GOOGLE_CLIENT_ID` isn't set in Render, or Render hasn't restarted since you added it. That's Part 7.1.

### Sign-in fails with "origin mismatch"

Your Vercel address isn't listed in Google's **Authorised JavaScript origins**. Add it exactly, with no trailing slash.

### The leaderboard empties every so often

Supabase isn't connected. That's Part 7.2 — until then, data lives only in memory and is lost whenever Render restarts.

### Uploading a PDF gives an error

If the PDF is a scan or photo of a page, there's no text inside it for the app to read. Use a PDF that you can select text in.

---

# Changing things later

**To change any setting** (AI key, database, sign-in): Render → your service → **Environment** → edit → **Save Changes**. Never edit these in files.

**To change the app's text or wording:** GitHub → find the file → pencil icon → edit → **Commit changes**. Vercel republishes automatically in about a minute.

**To undo a change that broke something:** GitHub → **Commits** (the clock icon above your file list) → find the last version that worked → **Revert**.

---

# What each secret setting does

Keep this for reference. All of these live in Render → Environment, never in a file.

| Setting | Required? | What it's for |
|---|---|---|
| `OPENROUTER_API_KEY` | **Yes** | Lets the app talk to an AI |
| `ALLOWED_ORIGINS` | **Yes** | Stops other websites using your AI key |
| `GOOGLE_CLIENT_ID` | Optional | Turns on Google sign-in |
| `SUPABASE_URL` | Optional | Where saved progress is stored |
| `SUPABASE_KEY` | Optional | Password for the above (`anon public` only) |
| `PUBLIC_APP_URL` | Optional | Your app's address, used for AI-service attribution |
