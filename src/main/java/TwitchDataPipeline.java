import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class TwitchDataPipeline {

    // Dataset coverage start (UTC): 2021-07-31 00:00:00
    // Epoch seconds for 2021-07-31 00:00:00 = 1627689600
    // Used for viewer timestamps
    private static final long START_EPOCH_UTC = 1627689600L;

    public static void main(String[] args) {
        // cmd-line args
        if (args.length != 1) {
            System.err.println("Usage: TwitchDataPipeline-1.0 <input_file>");
            System.exit(1);
        }

        String inputFile = args[0];

        // Spark session
        SparkSession spark = SparkSession.builder()
            .appName("TwitchDataPipeline")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate();

        // Define data schema since Kaggle dataset doesn't include headers
        StructType schema = new StructType()
            .add("user_id", DataTypes.StringType)
            .add("stream_id", DataTypes.StringType)
            .add("streamer_username", DataTypes.StringType)
            .add("time_start", DataTypes.LongType)
            .add("time_stop", DataTypes.LongType);

        // Load initial raw data
        Dataset<Row> rawData = spark
            .read()
            .option("header", "false")
            .schema(schema)
            .csv(inputFile);

        // Drop any rows with missing values
        Dataset<Row> cleanData = rawData.na().drop();

        // Expand time buckets by creating a sequence of time buckets, so if user has start time 1 and
        // and stop time 3, the sequence will be [1, 2, 3]
        Dataset<Row> expandedData = cleanData.withColumn(
            "time_buckets",
            sequence(col("time_start"), col("time_stop"))
        );

        // Explode the time_buckets column to create individual time buckets, so each
        // of [1, 2, 3] would be its own row entry, as a result we will only have
        // 4 columns as time_start and time_stop would be replaced with time_bucket
        Dataset<Row> individualBuckets = expandedData
            .withColumn("time_bucket", explode(col("time_buckets")))
            .select(
                col("user_id"),
                col("stream_id"),
                col("streamer_username"),
                col("time_bucket")
            );

        // individualBuckets.show();
    }
}
