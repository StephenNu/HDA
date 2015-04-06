package weka.filters;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.*;
import weka.filters.SimpleBatchFilter;

import java.util.ArrayList;
import java.util.HashMap;

import weka.core.Instances;
import weka.core.TestInstances;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;
import weka.filters.HDA;

import junit.framework.Test;
import junit.framework.TestSuite;

public class HDATest
  extends AbstractFilterTest {

  private static final double DELTA = 1e-15;

  public HDATest(String name) {
    super(name);
  }

  protected void setUp() throws Exception {
    super.setUp();

    m_Instances.deleteAttributeAt(6);  // Attribute Numeric type missing values
    //m_Instances.deleteAttributeAt(5);// Attribute Date type missing values
    m_Instances.deleteAttributeAt(4);  // Attribute Nominal type missing values
    m_Instances.deleteAttributeAt(3);  // Attribute String type missing values
    //m_Instances.deleteAttributeAt(2);// Attribute Numeric Type
    m_Instances.deleteAttributeAt(1);  // Attribute Nominal type
    m_Instances.deleteAttributeAt(0);  // Attribute String type

    // Half to class one, half to class two.
    for (int i = 0; i < m_Instances.numInstances(); ++i) {
      if (i < m_Instances.numInstances()/2) {
        m_Instances.instance(i).setValue(1, 0);
      } else {
        m_Instances.instance(i).setValue(1, 1);
      }
    }
    m_Instances.setClassIndex(1);
  }

  /** 
   * @return Creates a default HDA filter.
   */
  public Filter getFilter() {
    return new HDA();
  }

  /**
   * Gets the data after being filtered.
   *
   * @return              The dataset has been reduced in dimensions.
   * @throws Exception    An exception is thrown when the data generation fails
  protected Instances getFilterClassiferData() throws Exception {
    System.out.println("Class index " + m_Instances.instance(0).classIndex());
    m_Instances.setClassIndex(m_Instances.numAttributes()-1);
    return m_Instances;
  }

   */

  // Place tests here of the form
  // public void test<test-name>() { ... test code ... }


  public void testSeparateDatasetByClass() {
    final int NUM_CLASSES = 4;
    final int NUM_INSTANCES = 5;

    ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
    attinfo.add(new Attribute("example-ID"));
    attinfo.add(new Attribute("class-ID"));
    Instances testInst = new Instances("Test instances", attinfo, 0);
    testInst.setClassIndex(1);
    int id = 0;
    // Set up NUM_CLASSES with NUM_INSTANCES each with unique ID
    for (int i = 0; i < NUM_CLASSES; ++i) {
      for (int j = 0; j < NUM_INSTANCES; ++j ) {
        Instance inst = new DenseInstance(2);
        inst.setValue(0, id);
        inst.setValue(1, i);
        ++id;
        testInst.add(inst);
      }
    }

    // Process the instances.
    HDA filter = (HDA)getFilter();
    HashMap<Integer, Instances> disjointDataset
            = filter.separateDatasetByClass(testInst);
    HashMap<Integer, ArrayList<Integer>> actual_separation
            = new HashMap<Integer, ArrayList<Integer>>();

    // Check actual output equals expected output.
    for (int i = 0; i < NUM_CLASSES; ++i) {
      assertTrue(disjointDataset.containsKey(i));
      actual_separation.put(i, new ArrayList<Integer>());
      for (int j = 0; j < NUM_INSTANCES; ++j) {
        actual_separation.get(i).add(
                (int)disjointDataset.get(i).instance(j).value(0)
        );
      }
    }
    int expected_id = 0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
      ArrayList<Integer> list = actual_separation.get(i);
      for (int j = 0; j < NUM_INSTANCES; ++j) {
        assertTrue(list.contains(expected_id));
        ++expected_id;
      }
    }
  }

  public static Test suite() {
    return new TestSuite(HDATest.class);
  }

  public static void main(String[] args) {
    junit.textui.TestRunner.run(suite());
  }
}
