package RBFModel_0328
// Radial basic Function Model (Radyal temel fonksiyon modeli)
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RBFModel_0328.{RBFRegression, RBFRegressionModel}
import org.apache.spark.sql.SparkSession

object RBFN  extends App {
  val sparks = SparkSession
    .builder()
    .appName("Spark SQL basic example")
    .config("spark.master", "local")
    .getOrCreate()
  val conf = new SparkConf()
    .setAppName("SparkMe Application")
    .setMaster("local[*]")  // local mode

  // 2. Create Spark context
  //val sc = new SparkContext(conf)
  //val txt2 = sparks.read.format("csv").option("header", "false").option("inferSchema", "true").option("delimiter", " ").load("/lokasyon")
  val txt2 = sparks.read.format("csv") // egitim dosyasinin okunmasi
    .option("header", "true")
    .option("inferSchema", "true")
    .option("delimiter", " ")
    .load("lokasyon")
  txt2.printSchema()
  //sparks.udf.register("toVector", (x: Double, y: Double, z:Double) => Vectors.dense(x,y,z))
  //val features = txt2.selectExpr("toVector(x, y, z) as features")
  //val feature = txt2.selectExpr("toVector(x, y) as feature")

  val featureAssembler = new VectorAssembler() // x y giris verileriyle features sutunun olusturulmasi
    .setInputCols(Array("x", "y"))
    .setOutputCol("features")

  // features.show() // features ıs a DataFrame
  val rbfr = new RBFRegression("").setFeaturesCol("features").setLabelCol("z") // bu sutunlar ile z koordinatinin alinmasi
    .setNumGauss(200).setWidthFactor(3).setSeed(1995)

  val pipeline = new Pipeline().setStages(Array(
    featureAssembler, rbfr))

  val tm = System.nanoTime()

  val rbfrModel = pipeline.fit(txt2)
  //val trainingSummary = lrModel

  val duration = (System.nanoTime() - tm) / 1e9d // calisma suresi hesaplanir

  println("training done")
  println("Duration Time =  " + duration + "  seconds")

  // read the resampling data
  val resample_data = sparks.read.format("csv") 
    .option("header", "false")
    .option("inferSchema", "true")
    .option("delimiter", " ")
    .load("/hlokasyon")
    .withColumnRenamed("_c0","x")
    .withColumnRenamed("_c1","y")
    .withColumnRenamed("_c2","z")
  //sparks.udf.register("toVector", (x: Double, y: Double, z:Double) => Vectors.dense(x,y,z))
  resample_data.printSchema()
  resample_data.show()
  val tm_r = System.nanoTime()
  val renamed_resample_data = featureAssembler.transform(resample_data)
  renamed_resample_data.printSchema()
  renamed_resample_data.show()

  val duration_r = (System.nanoTime() - tm_r) / 1e9d

  println("resapling done")
  println("Duration Time =  " + duration_r + "  seconds")
  val holdout  = rbfrModel.transform(resample_data).select("x" , "y", "prediction") // veriler ile rbf modelin olusturulmasi
  //holdout.printSchema()
  holdout.show()
  holdout.write.format("com.databricks.spark.csv").mode("overwrite").option("delimiter"," ").save("/lokasyon")
  // modelin kaydedilmesi

  // save the model on a file
  rbfrModel.stages(1).asInstanceOf[RBFRegressionModel].write.overwrite().save("/lokasyon")

  rbfrModel.stages(1).asInstanceOf[RBFRegressionModel].getSigma().foreach(x => println(x))
  rbfrModel.stages(1).asInstanceOf[RBFRegressionModel].getCenters().foreach(x => println(x))

  // cre
  val rbf2 = RBFRegressionModel.load("/lokasyon")
  rbf2.getSigma().foreach(x => println(x))
  rbf2.getCenters().foreach(x => println(x))
  //rbfrModel.save("myrbf.txt")
  //println(rbfrModel.stages(1).asInstanceOf[RBFRegressionModel].getSigma().mkString(" "))
  //println(rbfrModel.stages(1).asInstanceOf[RBFRegressionModel].getCoefficients().mkString(" "))
  //println(rbfrModel.stages(1).asInstanceOf[RBFRegressionModel].getIntercept())
  // println(renamed_resample_data.select("features").foreach(r => rbfrModel.stages(1).asInstanceOf[RBFRegressionModel].predictWhere(r.getAs[linalg.Vector]("features"))))

  // son olarak bu kaydedilen veri dosyasi matlab programi ile acilarak grafik uzerinde incelendi
}
