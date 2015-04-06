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

import weka.core.Instances;
import weka.core.TestInstances;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;

import junit.framework.Test;
import junit.framework.TestSuite;

public class HDATest
  extends AbstractFilterTest {

  public HDATest(String name) {
    super(name);
  }

  protected void setUp() throws Exception {
    super.setUp();
    
    m_Instances.deleteAttributeAt(6);
    //m_Instances.deleteAttributeAt(5);
    m_Instances.deleteAttributeAt(4);
    m_Instances.deleteAttributeAt(3);
    //m_Instances.deleteAttributeAt(2);
    m_Instances.deleteAttributeAt(1);
    m_Instances.deleteAttributeAt(0);
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
  public void testSanity() {
    System.out.println("Hello tests " + m_Instances.instance(0).classIndex());
    m_Instances.setClassIndex(m_Instances.numAttributes()-1);
    System.out.println("Hello tests " + m_Instances.instance(0).classIndex());
    System.out.println(m_Instances.numInstances());
    System.out.println(m_Instances.size());
    assertEquals(1, 1);
  }
  public static Test suite() {
    return new TestSuite(HDATest.class);
  }

  public static void main(String[] args) {
    junit.textui.TestRunner.run(suite());
  }
}
