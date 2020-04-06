NSAMPLES=1000
METHOD=Latin
JAVA_ARGS="-cp MOEAFramework-2.4-Demo.jar"

# Generate the parameter samples
echo -n "Generating parameter samples..."
java ${JAVA_ARGS} \
    org.moeaframework.analysis.sensitivity.SampleGenerator \
    --method ${METHOD} --n ${NSAMPLES} --p uncertain_params_original.txt \
    --o LHsamples_original_1000.txt
java ${JAVA_ARGS} \
    org.moeaframework.analysis.sensitivity.SampleGenerator \
    --method ${METHOD} --n ${NSAMPLES} --p uncertain_params_narrowed.txt \
    --o LHsamples_narrowed_1000.txt

