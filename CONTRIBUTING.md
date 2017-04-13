# Contributing to HClib

Before opening a Pull Request,
please be sure to read through these guidelines.
Following the guidelines listed below will make it much more likely
for your contributions to be quickly accepted and merged.

## Licensing

All contributions submitted to this project are governed
by the terms outlined in our license agreement.
Specifically, our license (Apache 2.0)
has an explicit contributions clause in [section
5](https://github.com/habanero-rice/hclib/blob/6ad778a/LICENSE#L131-L137):

> Unless You explicitly state otherwise, any Contribution intentionally
> submitted for inclusion in the Work by You to the Licensor shall be under
> the terms and conditions of this License, without any additional terms or
> conditions.  Notwithstanding the above, nothing herein shall supersede or
> modify the terms of any separate license agreement you may have executed
> with Licensor regarding such Contributions.

## Pull Requests

Please follow established best practices for your pull requests.
Give a short description of *what* your Pull Request does in the title,
and include a short description of *why* the Pull Request is needed in the comment.
A complex title or summary is a good indication that your Pull Request
is doing too many things, and should be split into multiple Pull Requests.
In general, your Pull Request should be doing a single thing whenever possible.

Your Pull Request should be squashed to a single commit, in the spirit of making small,
incremental, and separable changes, and to simplify the log. Please do not squash your unrelated commits just to 
comply with this request, if you do so you will be asked to separate your changes. Your commit message should
explain *what* and *why* your commit does, not *how*. The *how* should be explained in your code comments.

When you create a Pull Request, please assign one or more 
[`core team members`](https://github.com/orgs/habanero-rice/teams/habanero-rice-devs) 
as the reviewers. 

Once you create a Pull Request, GitHub will automatically start a discussion 
concerning your request. You should monitor the discussion and address the suggestions,
concerns, and critiques from other collaborators.

### Merging Pull Requests

Only the 
[`Core team members`](https://github.com/orgs/habanero-rice/teams/habanero-rice-devs)
have merge privileges for Pull Requests in the master branch, but all collaborators
are welcome to review and discuss a particular Pull Request. 

Every Pull Request must be signed off by two Core team members in order to be merged. After one Core member completes their review and is satisfied that all their comments are addressed they should leave a comment indicating their approval. After a second Core member completes a review and had all comments addressed, they may perform the actual merge.

Core members should **never** merge their own Pull Requests, they must follow the same guidelines above as all the other contributors and have their Pull Request reviewed and merged by two other Core members.


## Code Review Guidelines
Here's a nice set of 
guidelines to follow when reviewing and discussing contributor's code (borrowed heavily and modified from 
[`Thoughtbot`](https://github.com/thoughtbot/guides/tree/master/code-review)
): 

### Everyone

* Accept that many programming decisions are opinions. Discuss tradeoffs, which
  you prefer, and reach a resolution quickly.
* Ask good questions; don't make demands. ("What do you think about naming this
  `:user_id`?")
* Avoid judgment and avoid assumptions about the author's
  perspective.
* Ask for clarification. ("I didn't understand. Can you clarify?")
* Avoid selective ownership of code. ("mine", "not mine", "yours")
* Avoid using terms that could be seen as referring to personal traits. ("dumb",
  "stupid"). Assume everyone is intelligent and well-meaning.
* Be explicit. Remember people don't always understand your intentions online.
* Be humble. ("I'm not sure - let's look it up.")
* Don't use hyperbole. ("always", "never", "endlessly", "nothing")
* Don't use sarcasm.
* Keep it real. If emoji, animated gifs, or humor aren't you, don't force them.
  If they are, use them with aplomb.
* Talk synchronously (e.g. chat, screensharing, in person) if there are too many
  "I didn't understand" or "Alternative solution:" comments. Post a follow-up
  comment summarizing the discussion.

### Having Your Code Reviewed

* Be grateful for the reviewer's suggestions. ("Good call. I'll make that
  change.")
* Don't take it personally. The review is of the code, not you. Be aware of how hard it is to convey 
  emotion online and how easy it is to misinterpret feedback. 
* Keping the previous point in mind: assume the best intention from the reviewer's comments.
* Explain why the code exists. ("It's like that because of these reasons. Would
  it be more clear if I rename this class/file/method/variable?")
* Seek to understand the reviewer's perspective.
* Try to respond to every comment.

### Reviewing Code

Understand why the change is necessary (fixes a bug, improves the user
experience, refactors the existing code). Then:

* Communicate which ideas you feel strongly about and those you don't.
* Identify ways to simplify the code while still solving the problem.
* If discussions turn too philosophical or academic, move the discussion offline
* Offer alternative implementations, but assume the author already considered
  them. ("What do you think about using an off the shelf library here?")
* Seek to understand the author's perspective.
* Sign off on the Pull Request with a :thumbsup: or "Ready to merge" comment.

## Quality Control for Contributed Code

### Regression Tests

The main regression tests for HClib are located in `test/c` and `test/cpp`.
The `test_all.sh` scripts in each of those directories will automatically
build and run all tests in the respective test suite.

All new commits should pass all of the regression tests.
When you open a Pull Request, Travis-CI will automatically run
the full regression suite against your cumulative change
(not against individual commits).
Please run the tests locally against each commit to ensure
that intermediate commits do not fail any tests
(which breaks the ability to use `git bisect` and similar tools).

If your Pull Request fixes a bug, it should also include a regression test for detecting that bug (with the appropriate changes in `test/c`, `test/cpp` and `test_all.sh`. This is to reduce the possibility of reintroducing the bug with some later changes.

If your Pull Request implements a new feature, it should also include tests in `test/c` and `test/cpp` that test that feature.

### Static Error Checking

As part of the development workflow for HClib, any newly committed code
should be checked using standard static checking tools.

In particular, run
[`cppcheck`](https://sourceforge.net/projects/cppcheck/)
on all modified files.
cppcheck should be run by `cd`-ing to `tools/cppcheck` and executing
the `run.sh` script from there (this assumes `cppcheck` is on your path).
Any new errors printed by `cppcheck` should be addressed before committing.

### Code Formatting

You should also run `astyle` on all modified or newly created files.
`astyle` is a source code auto-formatter.
Simply `cd` to `tools/astyle` and execute the `run.sh` script.
This assumes you have `astyle` installed and it is on your path.

## Summary

Here are some useful tips for making good Pull Requests (borrowed and heavily modified from 
[`Mark Seemann's blog`](http://blog.ploeh.dk/2015/01/15/10-tips-for-better-pull-requests/)
):

1. *Make it small.*
A small, focused Pull Request gives you the best chance of having it accepted. The time it takes to review a Pull Request 
is typically non-linear in the size of the request. Don't make your reviewers work.

2. *Do only one thing.*
Your Pull Request should address only a single concern. If it addresses multiple concerns (say A, B and C), the reviewers might 
be fine with A and B, but have issues with C, delaying the merge of your Pull Request. This might unecessarily complicate the merge 
of your perfectly fine A and B contributions if there were some updates to the repository in the meantime.

3. *Watch your line width.*
The reviewers will typically view the diffs side-by-side, and code and/or comments that are too wide will force them 
to scroll horizontally. Don't make your reviewers work. All newly merged code should be no wider than 80 columns.

4. *Avoid re-formatting.*
Please avoid re-formatting the existing code to fit 'your' style. All changes will show up in the diff, forcing your reviewers to 
decipher what the changes might be doing. Don't make your reviewers work. If you *must* make style changes (moving the code around, 
change formatting etc.), do it in a separate Pull Request that does only that and make it clear in the comment. Use `astyle` for the
code you are contributing.

5. *Make sure the code builds.*
Duh.

6. *Make sure all tests pass.*
Duh. 

7. *Add tests.*
If your contribution can be tested, it should be tested.

8. *Document your reasoning.*
You should always strive to write *self-documenting code* by using well-named operations, types and values in your code. 

When that is not sufficient (for example, if your code is making certain assumptions about the state of the program), 
add clear *comments* to your code explaining your reasoning.

Your *commit messages* should help the reviewers understand what are you doing and why are you doing it. The commit messages
will dissappear if we ever change the version control system, so don't rely on them exclusively to explain your code.

Your *Pull Request comments* should also help the reviewers understand why are you doing what you're doing, typically
in reference to something outside of the source code (other issues or bug reports). The Pull Request Comments 
will dissappear if we ever change the host so do not rely on them exclusively to explain your commit.

9. *Write well.*
Just as with your code, your prose (in comments, commit messages, Pull Request comments and Pull Request discussions should be 
clear and unambiguous. If your prose is hard to understand, you will make your reviewers work. Don't make your reviewers work.

10. *Squash your changes into a single commit.*
To avoid thrashing and reinforce Tips 1. and 2., your Pull Request should contain a single commit. When addressing the reviewer's 
comments and request for changes, do it by creating separate commits, so that the reviewers can see how you have addressed their comments.
Once you get a thumbs up from two Core Team reviewers, squash all your changes into a single commit.
You are the only owner of your Pull Request branch, so you can modify and force push it all you want. This will make the resulting commit history cleaner.

