# Job Script Templates

This directory contains a set of Slurm job scripts designed to be copied and modified by cluster users.  The general construction of the scripts is informed by our experiences with the templates produced for Mills and Farber.


## Keep the Templates Simple

The job script templates on previous clusters tended to include fairly complex Bash code that configured the job environment.  For example, the MPI scripts on Mills eventually included code that detected and adapted to the number of PSM contexts remaining on the nodes.  The plethora of boilerplate code inside the job script template made it difficult for users to know what to edit and where.  It also made it extremely difficult to fix or update the template behavior, since anyone who had made copies of the template would need to edit a new copy of the template.

On Caviness, the job script templates stick to setting environment variables to control how the runtime environment should be configured, and the Bash code to effect those settings has been moved out of the template.  You'll note that the templates *source* one or more external scripts:

```bash
#
# Do standard OpenMP environment setup:
#
. /opt/shared/slurm/templates/libexec/openmp.sh
```

In the future, IT staff can make changes to that external script and effectively augment all copies of the template that users have made.


### Example Scenario

OpenMP can be instructed to bind the threads it spawns to specific processor cores, to avoid the threads migrating between NUMA domains, etc.  The GNU and Intel runtime environments for OpenMP will automatically associate threads with available processor cores unless otherwise instructed.  Gaussian is compiled with the Portland compiler suite.  Long after the [generic/threads.qs](./generic/threads.qs) template was written the [applications/gaussian.qs](./applications/gaussian.qs) template was written and tested.  The Gaussian jobs failed with errors indicating the variable `MP_BLIST` was not set.

As it happens, the Portland runtime for OpenMP demands that explicit processor core bindings are provided that map threads to cores.  The list of processor indices is provided in the `MP_BLIST` environment variable.  We were able to alter the external script ([../libexec/openmp.sh](../libexec/openmp.sh)) to set `MP_BLIST` without having to alter [generic/threads.qs](./generic/threads.qs), [applications/gaussian.qs](./applications/gaussian.qs), and all copies that users had made.


## Logical Structure

The job script templates are all structured in sections.  The token "[EDIT]" is used to signal the user what areas can/should/must be altered in their copy of the template.  Sections occur in the following order:

1.  The hash-bang (start a bash shell in login mode)
2.  A lengthy comment section which includes Slurm SBATCH flags for job submission and discussion of their use
3.  VALET addition of software packages to the environment
4.  Setup of environment variables associated with the VALET software packages that were added
5.  Any variables that direct an external script's setup of the environment are defined and explained
6.  External scripts are sourced
7.  Any pre-execution setup actions occur
8.  The job is executed and its return code saved
9.  Any post-execution cleanup occurs
10. The saved return code is the script's return code


### Environment Variables

Environment variables that control the behavior of the external scripts sourced by the job templates MUST be prefixed with `UD_`.

Environment variables associated with a specific application SHOULD use a common prefix.  Examples are `GAUSSIAN_INPUT_FILE` and `GROMACS_MDRUN_FLAGS`.


### External Scripts

The external scripts that are sourced by the templates can be found in the [libexec](../libexec) directory in this tree.  The scripts will produce output (in the job's Slurm output file) describing the changes they make unless the `UD_QUIET_JOB_SETUP` environment variable is set.


## Structure of this Directory

The [jobscripts](./) directory is organized into:

- [applications](./applications) templates for specific programs that IT supports on the cluster
- [generic](./generic) templates that are the basis for [applications](./applications) templates

Using a template from the [generic](./generic) directory implies that the user will be required to make more extensive alterations to tailor it to his/her specific computational application.  While the [applications](./applications) templates may require edits by the user, they are much closer to a working job script and represent what IT staff deem a *best practice* for executing that application on the cluster.

Any edits introduced by the user into a copy of a template should try to adhere to the logical structure described in the previous section.
