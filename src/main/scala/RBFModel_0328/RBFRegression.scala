package  org.apache.spark.ml.regression
package RBFModel_0328

// RBF Modeli burada olusturuldu .
//LinearRegression modeli baz alindi
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.util._

import scala.collection.mutable
//import org.apache.spark.ml.util.{GeneralMLWritable, GeneralMLWriter}
//import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, MLWriter}
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}


class RBFRegression (override val uid: String) extends Regressor[Vector,RBFRegression,RBFRegressionModel] {

  def this() = this(Identifiable.randomUID("RBFReg"))

  var numGauss = 1
  var WidthFactor = 1.0
  var seed = 0.toLong

  override def train(dataset: Dataset[_]): RBFRegressionModel = {

    val clusterStage = new KMeans().setFeaturesCol(getFeaturesCol).setK(getNumGauss()).setSeed(getSeed())

    //    object Mytransformer extends Transformer {
    //      val uid = Identifiable.randomUID("MyT")
    //      override def transform(dataset: Dataset[_]): DataFrame = {
    //        val newColumn = udf((x: Double) => x.toString).apply(col("prediction"))
    //        dataset.withColumn("new", newColumn)
    //
    //        dataset.groupBy("prediction").agg(avg("x").as("centers"))
    //      }
    //      override def transformSchema(schema: StructType): StructType = {
    //        schema
    //      }
    //      override def copy(extra: ParamMap): Transformer = {
    //        defaultCopy(extra)
    //      }
    //    }

    val KMpipeline = new Pipeline()
      //      .setStages(Array(clusterStage,Mytransformer))
      .setStages(Array(clusterStage))


    // Fit the pipeline to training documents.
    val model = KMpipeline.fit(dataset) //pipelinemodel
    val transformed = model.transform(dataset)
    //transformed.show()

    object GaussCenterStage extends Transformer  {
      val uid = Identifiable.randomUID("GCEst")

      override def transform(dataset: Dataset[_]): DataFrame = {
        var centers = model.stages(0).asInstanceOf[KMeansModel].clusterCenters
        val centersColumn = udf((x: Int) => centers(x)).apply(col("prediction"))
        dataset.withColumn("centers", centersColumn)
          .withColumn("idG", col("prediction"))

      }
      override def transformSchema(schema: StructType): StructType = {
        schema.add(StructField("centers", DoubleType))
      }
      override def copy(extra: ParamMap): Transformer = {
        defaultCopy(extra)
      }
    }
    //GaussCenterStage.set("inputCol", "prediction")
    //   .set("outputCol", "centers")

    object GaussWidthStage extends Transformer {
      val uid = Identifiable.randomUID("GWEst")

      override def transform(dataset: Dataset[_]): DataFrame = {
        //val distance = udf(Vectors.sqdist _).apply(col("centerAndPoint"))
        //dataset.withColumn("CPArr", array(col("centers"), col("xyfeatures")))
        //val distance = udf((x : Array[Vector]) => Vectors.sqdist(x(1),x(2))).apply(col("CPArr"))
        val distance = udf((x : Vector, y:Vector) => Vectors.sqdist(x,y))
          .apply(col(getFeaturesCol),col("centers"))
        dataset.withColumn("distFromCenters", distance)

      }
      override def transformSchema(schema: StructType): StructType = {
        schema.add(StructField("distFromCenters", DoubleType))
      }
      override def copy(extra: ParamMap): Transformer = {
        defaultCopy(extra)
      }
    }
    //GaussWidthStage.set("inputCol", "centerAndPoint")
    object SigmaStage extends Estimator[RBFRegressionModel] {
      val uid = Identifiable.randomUID("Sigma")

      override def fit(dataset: Dataset[_]): RBFRegressionModel = {

        val sigma = dataset.groupBy("prediction")
          .agg(mean("distFromCenters").alias("sigmaSQ")
            //,mean(dataset.col(getLabelCol)).alias("temW") //  weıght estimate is here done as average z
          ).withColumn("sigma", udf((x: Double) => WidthFactor*math.sqrt(x)).apply(col("sigmaSQ")))

        //val sigmaColumn = dataset.join(sigma , sigma.col("prediction",") === dataset.col("prediction") , "inner").alias("sigma")
        val GparColumn = sigma.join(dataset.select("centers","prediction","idG"), Seq("prediction") , "inner").distinct()

        println("Gpar")
        //GparColumn.show()
        //dataset.show()

        val sigmaColumn = dataset.join(sigma , Seq("prediction") , "inner")
        val gaussian = udf((x : Double, y:Double) => math.exp(-(x)/ (y * y))).apply(col("distFromCenters"),col("sigma"))
        sigmaColumn.withColumn("GaussianExp", gaussian)

        println("sıgmaCol")
        //sigmaColumn.show()
        // add a column with the output of each Gaussian for each point

        //dataset.withColumn("id", monotonically_increasing_id)
        //GparColumn.withColumn("idG", monotonically_increasing_id)

        val datasetId =dataset.withColumn("id", monotonically_increasing_id).select("id", getFeaturesCol, getLabelCol)
        //val GparId = GparColumn.withColumn("idG", monotonically_increasing_id).select("idG","sigma","centers")
        val GparId = GparColumn.select("idG","sigma","centers")
        val weight = datasetId.crossJoin(GparId)
          .withColumn("distFromCenters", udf((x : Vector, y:Vector) => Vectors.sqdist(x,y))
            .apply(col(getFeaturesCol),col("centers")))
          //.withColumn("GOut",struct(gaussian,col("idG")))
          .withColumn("GOut",gaussian)


        // weight.show()

        val spark: SparkSession =
          SparkSession
            .builder()
            .appName("AppName")
            .config("spark.master", "local")
            .getOrCreate()
        import spark.implicits._


        val Temp = weight.select("id", "idG", "GOut")
        Temp.printSchema()
        //Temp.select("*").where(col("id") ===1 or col("id") === 2).show(20, false)

        val dfWithFeat = Temp
          .rdd
          .map(r => (r.getAs[Long]("id").intValue(), (r.getAs[Int]("idG").intValue(), r.getAs[Double]("GOut") )))
          .groupByKey()
          .map(r => LabeledPoint(r._1, Vectors.sparse(numGauss, r._2.toSeq)))
          .toDS().withColumnRenamed("label", "id")
          .join(datasetId.select("id",getLabelCol), Seq("id")).withColumnRenamed(getLabelCol,"label")
        dfWithFeat.printSchema


        println("dfWithFeat")
        // dfWithFeat.show(20, false)

        val lr = new LinearRegression().setFitIntercept(false)

        val lrMod = lr.fit(dfWithFeat)

        //val lrMod = lrm


        //val lrMod = new LinearRegression().fit(dfWithFeat) //.fitIntercept(false)
        //val lrMod = lr.train(dfWithFeat)

        //weight.select("id","idG","GOut").where("id=1 or id=2" ).show(20, false)



        println(lrMod.coefficients) // weights
        //val GArray = weight.select("*").foreach(x => weight.where(col("id") === x).orderBy("idG")
        //.agg(collect_list("GOut").as("GArrayOut")))
        //   foreach weight.id as ID
        //     weight.select(allthecols).where(ıd==ID).orderby(ıdG).agg(as an array)

        //.sort("idG").select("id","GOut").agg(collect_list("GOut").as("GArrayOut"))
        //val GArray = weight.sort("idG").select("id","GOut").groupBy("id").agg(collect_list("GOut").as("GArrayOut"))
        //GArray.select("id","GArrayOut").where("id=1 or id=2" ).show(20,false)
        //val GArray = weight.groupBy("id").sort("idG").select(collect_list(col( "GOut")))
        val weightDF = Seq.range(0, numGauss) .map(t => (t, lrMod.coefficients.apply(t))).toSeq.toDF("idG","weights")

        //val df = spark.sparkContext.parallelize(lrMod.coefficients  Seq.range(0,numGauss)).toDF("Tvalues","Pvalues")


        val GPtransformed=  GparColumn.join(weightDF, Seq("idG"))

        new RBFRegressionModel("",
          centers = GPtransformed.orderBy("prediction").select("centers").collect().map(_.getAs[Vector]("centers")),
          sigma = GPtransformed.orderBy("prediction").select("sigma").collect().map(_.getAs[Double]("sigma")),
          coefficients = GPtransformed.orderBy("prediction").select("weights").collect().map(_.getAs[Double]("weights")),
          lrMod.intercept
        )

        //dataset.withColumn("GOut", )
        //dataset.groupBy("prediction").agg(mean("distFromCenters").alias("sigma"))  // group for sigma
        //dataset.join(dataset,"prediction").groupBy("prediction").agg(mean("distFromCenters").alias("sigma"))

      } // SigmaStage.fıt

      override def transformSchema(schema: StructType): StructType = {
        schema.add(StructField("sigma", DoubleType))
        // schema
      }

      override def copy(extra: ParamMap): Estimator[RBFRegressionModel] = {
        defaultCopy(extra)
      }
    } // object SigmaStage

    val GPpipeline = new Pipeline()
      //      .setStages(Array(GaussCenterStage,CenterAndPointAssembler,GaussWidthStage))

      .setStages(Array(GaussCenterStage,GaussWidthStage, SigmaStage))

    val GPtransformed = GPpipeline.fit(model.transform(dataset))//.transform(model.transform(dataset))
    //println("GPTransformed")
    //GPtransformed.show()
    //val GPtransformed = GPpipeline.fit(model.transform(dataset)).asInstanceOf[RBFRegressionModel]
    GPtransformed.stages(2).asInstanceOf[RBFRegressionModel]


    // GPtransformed

    /* new RBFRegressionModel("",
       centers = GPtransformed.orderBy("prediction").select("centers").collect().map(_.getAs[Vector]("centers")),
       sigma = GPtransformed.orderBy("prediction").select("sigma").collect().map(_.getAs[Double]("sigma")),
       coefficients = GPtransformed.orderBy("prediction").select("weights").collect().map(_.getAs[Double]("weights")),

 //      coefficients = GPtransformed.orderBy("prediction").select("weights").collect().map(_.getAs[Double]("weights"))
     )*/
  } //  RBFRegression.train

  def getNumGauss (): Int = {
    numGauss
  }
  def setNumGauss (value: Int): RBFRegression= {
    numGauss = value
    this
  }
  def getSeed (): Long = {
    numGauss
  }
  def setSeed(value: Long): RBFRegression= {
    seed = value
    this
  }

  def getWidthFactor (): Double = {
    WidthFactor
  }
  def setWidthFactor (value: Double): RBFRegression= {
    WidthFactor = value
    this
  }

  override def copy(extra: ParamMap): RBFRegression = {
    defaultCopy(extra)
  }
}

class RBFRegressionModel private[ml] (override val uid: String,
                                      val centers: Array[Vector],
                                      val sigma : Array[Double],
                                      val coefficients: Array[Double],
                                      val intercept: Double//,
                                      // val scale: Double
                                     )
  extends RegressionModel[Vector, RBFRegressionModel] with MLWritable {
  override def predict(features: Vector): Double = {

    //val oldcenters : Array[OldVector] = centers.map(c => OldVectors.fromML(c))
    //new MLlibKMeansModel(oldcenters).predict(OldVectors.fromML(features))
    // TO DO: add check to the length of features (it must match the centers elements length)

    val rbfid  = (0 to sigma.length-1) //toParArray
    var sum = 0.0
    var count = 0.0
    //rbfid.par.foreach((idx: Int) => {
    rbfid.foreach((idx: Int) => {
      // create dataframe to compute the output of the i-th gaussian

      var g = coefficients.apply(idx)*math.exp(-Vectors.sqdist(features, centers.apply(idx))/ (sigma.apply(idx) * sigma.apply(idx)) )
      sum += g
      //count += 1 * math.pow(10,idx)
    })
    sum + intercept
    //count
  }

 // override def write : GeneralMLWriter = new GeneralMLWriter(this)
 override def write: MLWriter = new RBFRegressionModel.RBFRegressionModelWriter(this)

  override def copy(extra: ParamMap): RBFRegressionModel = {
    defaultCopy(extra)
  }

  def getSigma(): Array[Double] = {
    sigma
  }
  def getCenters(): Array[Vector] = {
    centers
  }
  def getCoefficients(): Array[Double] = {
    coefficients
  }
  def getIntercept(): Double = {
    intercept
  }


} // class RBFRegresionModel


object RBFRegressionModel extends MLReadable[RBFRegressionModel] {

  override def read: MLReader[RBFRegressionModel] = new RBFRegressionModelReader

   override def load(path: String): RBFRegressionModel = super.load(path)

  /** [[MLWriter]] instance for [[LinearRegressionModel]] */
  private[RBFRegressionModel] class RBFRegressionModelWriter(instance: RBFRegressionModel)
    extends MLWriter with Logging {

    private case class Data(intercept: Double, centers:Array[Vector], sigma: Array[Double] ,coefficients: Array[Double])

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: intercept, coefficients
      val data = Data(instance.intercept,instance.centers, instance.sigma ,instance.coefficients)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private[RBFRegressionModel] class RBFRegressionModelReader extends MLReader[RBFRegressionModel] {

        /** Checked against metadata when loading model */
        private val className = classOf[RBFRegressionModel].getName

        override def load(path: String): RBFRegressionModel = {
          val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

          val dataPath = new Path(path, "data").toString
          val data = sparkSession.read.format("parquet").load(dataPath)
          data.printSchema()
          val Row (intercept: Double, centers: mutable.WrappedArray[Vector], sigma: mutable.WrappedArray[Double], coefficients: mutable.WrappedArray[Double]) =
          data
                .select("intercept","centers","sigma", "coefficients")
            .head()


          val model = new RBFRegressionModel(metadata.uid, centers.toArray, sigma.toArray,  coefficients.toArray, intercept)

          DefaultParamsReader.getAndSetParams(model, metadata)
          model
        }
      }

}// object RBFRegressionModel extends MLReadable[RBFRegressionModel]
