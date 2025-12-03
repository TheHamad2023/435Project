import static org.apache.spark.sql.functions.*;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.apache.spark.sql.types.*;

import java.util.List;

public class TwitchDataPipeline {
    // Dataset start: 2021-07-31 00:00:00 UTC
    private static final long START_EPOCH_UTC = 1627689600L;
    // 600 -> 10 minutes
    private static final long SEC_PER_TICK = 600L;

    // Toggle: if true, materialize per-bucket rows from constant-count intervals.
    private static final boolean MATERIALIZE_PER_BUCKET = true;

    // Row limit
    // Dataset is too large, limit to 100 million rows
    private static final int ROW_LIMIT = 100000000;

    // Target top-32 streamers
    // Tens of thousands of low-to-now viewer streamers pollute data
    private static final int TOP_N_STREAMERS = 32;

    // Specifc streamer to compare Actual vs Predicted viewer count
    // Just used as an example case, does not effect the training of the model
    private static final String PLOT_STREAMER = "ninja";

    public static void main(String[] args) {
        String inputPath = args.length > 0 ? args[0] : "data/twitch.csv";
        String outDirRoot = args.length > 1 ? args[1] : "out";

        SparkSession spark = SparkSession.builder()
                .appName("TwitchDataPipeline")
                .config("spark.sql.session.timeZone", "UTC")
                .getOrCreate();

        // Create schema. Dataset is headerless
        StructType schema = new StructType()
                .add("user_id", DataTypes.StringType)
                .add("stream_id", DataTypes.StringType)
                .add("streamer_username", DataTypes.StringType)
                .add("time_start", DataTypes.LongType)
                .add("time_stop", DataTypes.LongType);

        // Read in raw data
        Dataset<Row> raw = spark.read()
                .option("header", "false")
                .schema(schema)
                .csv(inputPath)
                .limit(ROW_LIMIT);

        // Filter out bad data
        Dataset<Row> clean = raw
                .filter(col("streamer_username").isNotNull())
                .filter(col("time_start").isNotNull().and(col("time_stop").isNotNull()))
                .filter(col("time_start").geq(lit(0)))
                .filter(col("time_stop").geq(col("time_start")))
                .select("streamer_username", "time_start", "time_stop")
                .repartition(col("streamer_username"))
                .persist();

        // Filter top-n streamers
        Dataset<Row> topStreamers = clean.groupBy("streamer_username")
                .count()
                .orderBy(col("count").desc())
                .limit(TOP_N_STREAMERS)
                .select("streamer_username");

        // Collect top-n streamer usernames
        List<String> topUsernamesList = topStreamers.as(Encoders.STRING()).collectAsList();

        // Filter the main dataset to only include the top-n streamers
        // * if Object[] part isn't here compiler complains *
        clean = clean.filter(col("streamer_username").isin((Object[]) topUsernamesList.toArray(new String[0])));
        clean.unpersist();

        /*
         * Line-Sweep to get all time_slots without creating a row for every time.
         * Only happens if the flag is set. 
         */

        // start events
        Dataset<Row> starts = clean.select(
                col("streamer_username"),
                col("time_start").alias("tick"),
                lit(1).alias("delta"));

        // end events
        Dataset<Row> ends = clean.select(
                col("streamer_username"),
                col("time_stop").plus(lit(1)).alias("tick"),
                lit(-1).alias("delta"));

        // combine starts(+1) and ends(-1)
        Dataset<Row> deltas = starts.unionByName(ends);

        // group all events by streamer for same tick
        Dataset<Row> deltaByTick = deltas.groupBy("streamer_username", "tick")
                .agg(sum("delta").alias("delta"));

        // calculate cumulative viewer count
        WindowSpec cumW = Window.partitionBy("streamer_username")
                .orderBy("tick")
                .rowsBetween(Window.unboundedPreceding(), Window.currentRow());
        Dataset<Row> changePoints = deltaByTick
                .withColumn("viewer_count", sum(col("delta")).over(cumW))
                .orderBy(col("streamer_username"), col("tick"));

        // defining time intervals
        WindowSpec leadW = Window.partitionBy("streamer_username").orderBy("tick");
        Dataset<Row> intervals = changePoints
                .withColumn("next_tick", lead(col("tick"), 1).over(leadW))
                .filter(col("next_tick").isNotNull())
                .select("streamer_username", "tick", "next_tick", "viewer_count");

        // make sure there was atleast one active user
        Dataset<Row> activeIntervals = intervals.filter(col("viewer_count").gt(lit(0)));

       // if true -> just do simple explode
       // if not -> use line-sweep 
        Dataset<Row> perBucket;
        if (MATERIALIZE_PER_BUCKET) {
            perBucket = activeIntervals
                    .withColumn("time_bucket", explode(sequence(col("tick"), col("next_tick").minus(lit(1)))))
                    .select(
                            col("streamer_username"),
                            col("time_bucket").alias("tick"),
                            col("viewer_count"));
        } else {
            perBucket = activeIntervals
                    .withColumn("time_bucket", col("tick"))
                    .select(col("streamer_username"), col("time_bucket").alias("tick"), col("viewer_count"));
        }

        // converts the ticks into real timestamps
        Dataset<Row> viewers = perBucket
                .withColumn("bucket_epoch", lit(START_EPOCH_UTC).plus(col("tick").multiply(lit(SEC_PER_TICK))))
                .withColumn("bucket_time", from_unixtime(col("bucket_epoch")).cast("timestamp"))
                .repartition(col("streamer_username"))
                .persist();

        // Extracts information from data 
        Dataset<Row> feats = viewers
                .withColumn("hour_of_day", hour(col("bucket_time")))
                .withColumn("day_of_week", dayofweek(col("bucket_time")))
                .withColumn("is_weekend", when(col("day_of_week").isin(1, 7), lit(1)).otherwise(lit(0)));

        // looks back in time to past viewer counts and brings them into the current row
        // strongest predictor of future values are past values
        WindowSpec lagW = Window.partitionBy("streamer_username").orderBy("bucket_time");
        feats = feats
                .withColumn("lag_1", lag(col("viewer_count"), 1).over(lagW))
                .withColumn("lag_2", lag(col("viewer_count"), 2).over(lagW))
                .withColumn("lag_6", lag(col("viewer_count"), 6).over(lagW))
                .na().drop(new String[] { "lag_1", "lag_2", "lag_6" });

        // spark.ml algo doesn't like when we use strings here, index it to an int instead
        StringIndexer streamerIndexer = new StringIndexer()
                .setInputCol("streamer_username")
                .setOutputCol("streamer_index");

        // spark ml wants vector input
        String[] featureCols = new String[] { "streamer_index", "hour_of_day", "day_of_week", "is_weekend", "lag_1",
                "lag_2", "lag_6" };
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        // actual RF model
        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("viewer_count")
                .setFeaturesCol("features")
                .setNumTrees(120)
                .setMaxDepth(12)
                .setMaxBins(64);

        // spark ml pipeline definition
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { streamerIndexer, assembler, rf });

        // Split dataset by time.
        // First 80% = training data
        // Last 20% = testing
        double[] q = feats.stat().approxQuantile("bucket_epoch", new double[] { 0.8 }, 0.001);
        long cutoffEpoch = (long) q[0];

        Dataset<Row> train = feats.filter(col("bucket_epoch").lt(lit(cutoffEpoch)));
        Dataset<Row> test = feats.filter(col("bucket_epoch").geq(lit(cutoffEpoch)));

        // Training and Evaluations
        PipelineModel model = pipeline.fit(train);

        Dataset<Row> preds = model.transform(test)
                .select("streamer_username", "bucket_time", "viewer_count", "prediction");

        // Get MAE and R^2 evaluations
        RegressionEvaluator maeEval = new RegressionEvaluator()
                .setLabelCol("viewer_count").setPredictionCol("prediction").setMetricName("mae");
        RegressionEvaluator r2Eval = new RegressionEvaluator()
                .setLabelCol("viewer_count").setPredictionCol("prediction").setMetricName("r2");

        // Output MAE and R^2 evaluations
        System.out.println("MAE = " + maeEval.evaluate(preds));
        System.out.println("R^2 = " + r2Eval.evaluate(preds));

        // Output predictions for a single streamer for plotting
        preds.filter(col("streamer_username").equalTo(PLOT_STREAMER))
                .orderBy("bucket_time")
                .coalesce(1)
                .write()
                .option("header", "true")
                .csv(outDirRoot + "/predictions_for_plot_" + PLOT_STREAMER);

        spark.stop();
    }
}