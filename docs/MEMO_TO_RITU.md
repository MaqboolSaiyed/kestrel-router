**To:** Ritu Deshpande, Head of D2C Operations · **Cc:** Farhan Sheikh, Meenal Joshi, Tanmay Kulkarni
**From:** Kabir Nanda's team · **Re:** Replacing the routing bot

---

**The decision.** Switch the bot off and use the new router, but judge it on a different number from the one in
your email. The labels in the export are the bot's own choices, and when we checked them against where each
request was actually resolved, the bot was wrong on 23% of requests. A system that matches those labels at 90%
would simply be a copy of the bot, mistakes included. We built one to check: it matched the bot 96% of the time
and sent customers to the wrong team just as often. So we trained the new router on where requests really ended up.

**The number.** The new router sends **85%** of requests to the team that resolves them, against **77%** for the
bot. We tested it on three separate quarters and it scored between 84% and 86% each time. When it is confident,
which is 83% of requests, it is right 96.5% of the time. The rest are messages like "please call me about my purifier",
which no system can route because the customer hasn't said what they need. For those, the router says so and
suggests one question to ask first. It also fixes the two problems Meenal raised: customers who mention paying for
a repair or installation are now routed correctly 86% of the time (the bot managed 3%), and purifier breakdowns
reach Repairs 98% of the time (the bot managed 25%).

**The rupees.** A misrouted request costs about **Rs 698**: on average 1.4 transfers at Rs 305 each plus one extra
customer contact at Rs 260 (your policy's figures). At today's volume of about 8,700 requests a year:

| | Cost a year |
|---|---|
| Keep the bot (misroutes plus the Rs 3.2 lakh licence) | Rs 17.0 lakh |
| New router, routing everything | Rs 9.2 lakh |
| New router, asking first when unsure | **Rs 5.6 lakh** |

That is a saving of **Rs 7.8 to 11.4 lakh a year**. For Farhan, in writing: the router runs on an existing office
machine or a small server, with **no charge per request**. The only optional outside service writes the question
for unclear requests; we measured it at under Rs 1 a month. Without it, a standard question is used.

**For headcount.** Plan on where work actually lands, not on the bot's queues. Each month the bot sends Repairs
about 206 requests, but Repairs resolves 166; Billing gets 110 and resolves 85. Installs & Demo gets 77 and
resolves 107, Returns gets 84 and resolves 109. The full table is in the evidence report.

**What to do next week**

1. **Run the router alongside the bot for two weeks** on live requests, without acting on it. Compare each
   decision with where the request is closed. This confirms the 85% on your own traffic before anything changes.
2. **Agree the measure with us now:** the share of requests closed by the first team they reach, not the match
   with the bot's labels.
3. **Ask Meenal to approve the question** sent to customers whose request is unclear, and start it on chat and
   WhatsApp, where asking is cheapest.
4. **Hold the renewal notice until the two weeks are done,** then tell the vendor. The switch itself is one day's work.

**What could go wrong.** The 85% assumes new requests look like recent ones; the router can be retrained on each
month's closed requests in seconds. The larger saving assumes customers answer the question; the two-week run
will show how many do.
