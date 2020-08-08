// dependencies
import React, { useState } from 'react'
import axios from 'axios'

// components
import BehindTheScenes from './components/BehindTheScenes'
import Form from './components/Form'
import Prediction from './components/Prediction'

// images
import banner from './img/banner.jpg'

// styles
import './css/styles.css'

function App() {
    // state hook variables
    const [prediction, setPrediction] = useState(undefined)
    const [title, setTitle] = useState('title_mrs')
    const [firstName, setFirstName] = useState('Margaret ("Molly")')
    const [lastName, setLastName] = useState('Brown')
    const [age, setAge] = useState(44)
    const [gender, setGender] = useState('gender_female')
    const [passengerClass, setPassengerClass] = useState('passenger_class_1')
    const [fare, setFare] = useState(27)
    const [cabinLocation, setCabinLocation] = useState('deck_b')
    const [embarkation, setEmbarkation] = useState('embarked_c')
    const [withChildren, setWithChildren] = useState(false)
    const [withParents, setWithParents] = useState(false)
    const [withSpouse, setWithSpouse] = useState(false)
    const [withSiblings, setWithSiblings] = useState(false)
    const [transmitting, setTransmitting] = useState(false)

    const makePrediction = e => {
        // create payload
        const payload = {
            title,
            age,
            gender,
            passengerClass,
            fare,
            cabinLocation,
            embarkation,
            withChildren,
            withParents,
            withSpouse,
            withSiblings
        }

        // update state
        setTransmitting(true)

        // post payload
        axios.post('/api/predict', { ...payload })
            .then(res => {
                // update state
                setPrediction(res.data)
                setTransmitting(false)
            }).catch(err => console.log(err))
    }

    return (
        <div className="container">
            <div className="row">
                <div className="col s12 banner" style={{ backgroundImage: `url(${banner})` }}></div>
            </div>
            <Prediction prediction={prediction} />
            <Form
                title={title}
                setTitle={setTitle}
                firstName={firstName}
                setFirstName={setFirstName}
                lastName={lastName}
                setLastName={setLastName}
                age={age}
                setAge={setAge}
                gender={gender}
                setGender={setGender}
                passengerClass={passengerClass}
                setPassengerClass={setPassengerClass}
                fare={fare}
                setFare={setFare}
                cabinLocation={cabinLocation}
                setCabinLocation={setCabinLocation}
                embarkation={embarkation}
                setEmbarkation={setEmbarkation}
                withChildren={withChildren}
                setWithChildren={setWithChildren}
                withParents={withParents}
                setWithParents={setWithParents}
                withSpouse={withSpouse}
                setWithSpouse={setWithSpouse}
                withSiblings={withSiblings}
                setWithSiblings={setWithSiblings}
                transmitting={transmitting}
                makePrediction={makePrediction}
            />
            <BehindTheScenes />
        </div>
    )
}

export default App
