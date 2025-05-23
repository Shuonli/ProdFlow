#include <QA.C>

#include <fun4all/Fun4AllDstOutputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllOutputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4allraw/Fun4AllStreamingInputManager.h>
#include <fun4allraw/InputManagerType.h>
#include <fun4allraw/SingleGl1PoolInput.h>
#include <fun4allraw/SingleInttPoolInput.h>
#include <fun4allraw/SingleMicromegasPoolInput.h>
#include <fun4allraw/SingleMvtxPoolInput.h>
#include <fun4allraw/SingleTpcPoolInput.h>

#include <intt/InttOdbcQuery.h>

#include <phool/recoConsts.h>

#include <ffarawmodules/InttCheck.h>
#include <ffarawmodules/StreamingCheck.h>
#include <ffarawmodules/TpcCheck.h>

#include <ffamodules/HeadReco.h>
#include <ffamodules/FlagHandler.h>
#include <ffamodules/SyncReco.h>

R__LOAD_LIBRARY(libfun4all.so)
R__LOAD_LIBRARY(libffamodules.so)
R__LOAD_LIBRARY(libfun4allraw.so)
R__LOAD_LIBRARY(libffarawmodules.so)

R__LOAD_LIBRARY(libintt.so)
bool isGood(const string &infile);

void Fun4All_Stream_Combiner(int nEvents = 100,
			     int runnumber = 30117,
			     const string &type = "beam",
			     const string &outdir = "/sphenix/lustre01/sphnxpro/commissioning/slurp/tpccosmics/",
			     const int     neventsper=100,
                             const string &input_gl1file = "gl1.list",
                             const string &input_inttfile00 = "intt0.list",
                             const string &input_inttfile01 = "intt1.list",
                             const string &input_inttfile02 = "intt2.list",
                             const string &input_inttfile03 = "intt3.list",
                             const string &input_inttfile04 = "intt4.list",
                             const string &input_inttfile05 = "intt5.list",
                             const string &input_inttfile06 = "intt6.list",
                             const string &input_inttfile07 = "intt7.list",
                             const string &input_mvtxfile00 = "mvtx0.list",
                             const string &input_mvtxfile01 = "mvtx1.list",
                             const string &input_mvtxfile02 = "mvtx2.list",
                             const string &input_mvtxfile03 = "mvtx3.list",
                             const string &input_mvtxfile04 = "mvtx4.list",
                             const string &input_mvtxfile05 = "mvtx5.list",
                             const string &input_tpcfile00 = "tpc00.list",
                             const string &input_tpcfile01 = "tpc01.list",
                             const string &input_tpcfile02 = "tpc02.list",
                             const string &input_tpcfile03 = "tpc03.list",
                             const string &input_tpcfile04 = "tpc04.list",
                             const string &input_tpcfile05 = "tpc05.list",
                             const string &input_tpcfile06 = "tpc06.list",
                             const string &input_tpcfile07 = "tpc07.list",
                             const string &input_tpcfile08 = "tpc08.list",
                             const string &input_tpcfile09 = "tpc09.list",
                             const string &input_tpcfile10 = "tpc10.list",
                             const string &input_tpcfile11 = "tpc11.list",
                             const string &input_tpcfile12 = "tpc12.list",
                             const string &input_tpcfile13 = "tpc13.list",
                             const string &input_tpcfile14 = "tpc14.list",
                             const string &input_tpcfile15 = "tpc15.list",
                             const string &input_tpcfile16 = "tpc16.list",
                             const string &input_tpcfile17 = "tpc17.list",
                             const string &input_tpcfile18 = "tpc18.list",
                             const string &input_tpcfile19 = "tpc19.list",
                             const string &input_tpcfile20 = "tpc20.list",
                             const string &input_tpcfile21 = "tpc21.list",
                             const string &input_tpcfile22 = "tpc22.list",
                             const string &input_tpcfile23 = "tpc23.list",
                             const string &input_tpotfile = "tpot.list"
)
{
// GL1 which provides the beam clock reference (if we ran with GL1)
  vector<string> gl1_infile;
  gl1_infile.push_back(input_gl1file);

// MVTX
  vector<string> mvtx_infile;
  mvtx_infile.push_back(input_mvtxfile00);
  mvtx_infile.push_back(input_mvtxfile01);
  mvtx_infile.push_back(input_mvtxfile02);
  mvtx_infile.push_back(input_mvtxfile03);
  mvtx_infile.push_back(input_mvtxfile04);
  mvtx_infile.push_back(input_mvtxfile05);

// INTT
  vector<string> intt_infile;
  intt_infile.push_back(input_inttfile00);
  intt_infile.push_back(input_inttfile01);
  intt_infile.push_back(input_inttfile02);
  intt_infile.push_back(input_inttfile03);
  intt_infile.push_back(input_inttfile04);
  intt_infile.push_back(input_inttfile05);
  intt_infile.push_back(input_inttfile06);
  intt_infile.push_back(input_inttfile07);

  vector<string> tpc_infile;
  tpc_infile.push_back(input_tpcfile00);
  tpc_infile.push_back(input_tpcfile01);
  tpc_infile.push_back(input_tpcfile02);
  tpc_infile.push_back(input_tpcfile03);
  tpc_infile.push_back(input_tpcfile04);
  tpc_infile.push_back(input_tpcfile05);
  tpc_infile.push_back(input_tpcfile06);
  tpc_infile.push_back(input_tpcfile07);
  tpc_infile.push_back(input_tpcfile08);
  tpc_infile.push_back(input_tpcfile09);
  tpc_infile.push_back(input_tpcfile10);
  tpc_infile.push_back(input_tpcfile11);
  tpc_infile.push_back(input_tpcfile12);
  tpc_infile.push_back(input_tpcfile13);
  tpc_infile.push_back(input_tpcfile14);
  tpc_infile.push_back(input_tpcfile15);
  tpc_infile.push_back(input_tpcfile16);
  tpc_infile.push_back(input_tpcfile17);
  tpc_infile.push_back(input_tpcfile18);
  tpc_infile.push_back(input_tpcfile19);
  tpc_infile.push_back(input_tpcfile20);
  tpc_infile.push_back(input_tpcfile21);
  tpc_infile.push_back(input_tpcfile22);
  tpc_infile.push_back(input_tpcfile23);

// TPOT
  vector<string> tpot_infile;
  tpot_infile.push_back(input_tpotfile);

  Fun4AllServer *se = Fun4AllServer::instance();
  se->Verbosity(0);
  recoConsts *rc = recoConsts::instance();
  // rc->set_IntFlag("RUNNUMBER",20445);
  Fun4AllStreamingInputManager *in = new Fun4AllStreamingInputManager("Comb");
  //  in->Verbosity(2);
// create and register input managers
  int NumInputs = 0;
  int i = 0;

  for (auto iter : gl1_infile)
  {
    if (isGood(iter))
    {
      SingleGl1PoolInput *gl1_sngl = new SingleGl1PoolInput("GL1_" + to_string(i));
      //    gl1_sngl->Verbosity(3);
      gl1_sngl->AddListFile(iter);
      in->registerStreamingInput(gl1_sngl, InputManagerType::GL1);
      i++;
    }
  }
  NumInputs += i;

  i = 0;
  for (auto iter : intt_infile)
  {
    if (isGood(iter))
    {
    SingleInttPoolInput *intt_sngl = new SingleInttPoolInput("INTT_" + to_string(i));
    //    intt_sngl->Verbosity(3);
    InttOdbcQuery query;
    bool isStreaming = true;
    if(runnumber != 0)
      {
	query.Query(runnumber);
	isStreaming = query.IsStreaming();
      }
    intt_sngl->streamingMode(isStreaming);
    
    intt_sngl->AddListFile(iter);
    in->registerStreamingInput(intt_sngl, InputManagerType::INTT);
    i++;
    }
  }
  NumInputs += i;

  i = 0;
  for (auto iter : mvtx_infile)
  {
    if (isGood(iter))
    {
    SingleMvtxPoolInput *mvtx_sngl = new SingleMvtxPoolInput("MVTX_" + to_string(i));
    //    mvtx_sngl->Verbosity(3);
    mvtx_sngl->AddListFile(iter);
    in->registerStreamingInput(mvtx_sngl, InputManagerType::MVTX);
    i++;
    }
  }
  NumInputs += i;

  i = 0;
  for (auto iter : tpc_infile)
  {
    if (isGood(iter))
    {
    SingleTpcPoolInput *tpc_sngl = new SingleTpcPoolInput("TPC_" + to_string(i));
//    tpc_sngl->Verbosity(2);
    //   tpc_sngl->DryRun();
    tpc_sngl->SetBcoRange(5);
    //tpc_sngl->SetMaxTpcTimeSamples(1024);
    tpc_sngl->AddListFile(iter);
    in->registerStreamingInput(tpc_sngl, InputManagerType::TPC);
    i++;
    }
  }
  NumInputs += i;

  i = 0;
  for (auto iter : tpot_infile)
  {
    if (isGood(iter))
    {
    SingleMicromegasPoolInput *mm_sngl = new SingleMicromegasPoolInput("MICROMEGAS_" + to_string(i));
    //   sngl->Verbosity(3);
    mm_sngl->SetBcoRange(10);
    mm_sngl->SetNegativeBco(2);
    mm_sngl->SetBcoPoolSize(50);
    mm_sngl->AddListFile(iter);
    in->registerStreamingInput(mm_sngl, InputManagerType::MICROMEGAS);
    i++;
    }
  }
  NumInputs += i;

// if there is no input manager this macro will still run - so just quit here
  if (NumInputs == 0)
  {
    std::cout << "no file lists no input manager registered, quitting" << std::endl;
    gSystem->Exit(1);
  }

  se->registerInputManager(in);
//  StreamingCheck *scheck = new StreamingCheck();
//  scheck->SetTpcBcoRange(130);
  // se->registerSubsystem(scheck);
  // TpcCheck *tpccheck = new TpcCheck();
  // tpccheck->Verbosity(3);
  // tpccheck->SetBcoRange(130);
  // se->registerSubsystem(tpccheck);

  SyncReco *sync = new SyncReco();
  se->registerSubsystem(sync);

  HeadReco *head = new HeadReco();
  se->registerSubsystem(head);

  FlagHandler *flag = new FlagHandler();
  se->registerSubsystem(flag);

  string outfilename = "./" + type + ".root";
  Fun4AllOutputManager *out = new Fun4AllDstOutputManager("out",outfilename);
  out->UseFileRule();
  out->SetNEvents(neventsper);                       // number of events per output file
  out->SetClosingScript("stageout.sh");      // script to call on file close (not quite working yet...)
  out->SetClosingScriptArgs(outdir);  // additional beyond the name of the file
  se->registerOutputManager(out);

  if (nEvents < 0)
  {
    return;
  }
  se->run(nEvents);

  se->End();
  TString qaname = "HIST_" + type + "-" + runnumber + "-0000.root";
  std::string qaOutputFileName(qaname.Data());
  QAHistManagerDef::saveQARootFile(qaOutputFileName);

  delete se;
  cout << "all done" << endl;
  gSystem->Exit(0);
}

bool isGood(const string &infile)
{
  ifstream intest;
  intest.open(infile);
  bool goodfile = false;
  if (intest.is_open())
  {
    if (intest.peek() != std::ifstream::traits_type::eof()) // is it non zero?
    {
      goodfile = true;
    }
      intest.close();
  }
  return goodfile;
}
